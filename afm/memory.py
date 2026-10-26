from dataclasses import dataclass


class MemoryGuardError(ValueError):
    pass


@dataclass(frozen=True)
class ChunkPlan:
    head_chunk_size: int
    query_chunk_size: int
    estimated_bytes: int


def plan_chunks(shape, element_size, layout, budget_mib, head_chunk_size, query_chunk_size):
    batch, heads, tokens, dim = shape
    if tokens != layout.total_tokens or min(*shape, element_size, head_chunk_size,
                                           query_chunk_size, budget_mib) <= 0:
        raise ValueError("Invalid attention shape or workspace budget")
    hc = min(heads, head_chunk_size)
    qc = min(layout.image_tokens, query_chunk_size)
    budget = int(budget_mib * 1024**2)

    def estimate(h, q):
        # Retained QKV + original/candidate output + full selected-row output.
        retained = batch * heads * tokens * dim * element_size * 6
        image_result = batch * heads * layout.image_tokens * dim * element_size
        # One batch row at a time. Includes FP32 copies, FFT complex scratch,
        # filter output, logits/probabilities and a conservative 25% margin.
        fp32_inputs = h * (layout.image_tokens + 2 * tokens) * dim * 4
        spectral = h * layout.image_tokens * layout.text_tokens * 40
        joint = h * q * (tokens * 8 + dim * 4)
        masks = layout.image_tokens * 24
        return int((retained + image_result + fp32_inputs + spectral + joint + masks) * 1.25)

    while estimate(hc, qc) > budget:
        choices = []
        if hc > 1:
            choices.append((estimate(max(1, hc // 2), qc), max(1, hc // 2), qc))
        if qc > 1:
            choices.append((estimate(hc, max(1, qc // 2)), hc, max(1, qc // 2)))
        if not choices:
            raise MemoryGuardError("memory_guard: minimum AFM workspace exceeds max_workspace_mib")
        _, hc, qc = min(choices)
    return ChunkPlan(hc, qc, estimate(hc, qc))
