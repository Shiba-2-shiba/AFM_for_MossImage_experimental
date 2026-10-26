import math

import torch

from .spectral import filter_logits, schedule_alphas


def edit_image_rows(q, k, v, layout, config, progress, chunks, scale):
    if (q.ndim != 4 or q.shape != k.shape or q.shape != v.shape
            or q.shape[2] != layout.total_tokens or q.device != k.device or q.device != v.device
            or q.dtype != k.dtype or q.dtype != v.dtype or not q.is_floating_point()):
        raise ValueError("Expected same-device/dtype [B,heads,I+T,d] native QKV")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Attention scale must be positive and finite")
    if not all(torch.isfinite(x).all() for x in (q, k, v)):
        raise FloatingPointError("Nonfinite QKV")
    batch, heads, _, dim = q.shape
    images = layout.image_tokens
    alpha_lf, alpha_hf = schedule_alphas(config.strength, progress)
    result = torch.empty((batch, heads, images, dim), dtype=v.dtype, device=v.device)
    with torch.autocast(device_type=q.device.type, enabled=False):
        for row in range(batch):
            for start in range(0, heads, chunks.head_chunk_size):
                end = min(start + chunks.head_chunk_size, heads)
                qi = q[row:row+1, start:end, :images].float()
                kh = k[row:row+1, start:end].float()
                vh = v[row:row+1, start:end].float()
                text_logits = (qi @ kh[:, :, images:].transpose(-2, -1)) * scale
                edited = filter_logits(text_logits, layout.grid, alpha_lf, alpha_hf,
                                       config.cutoff, config.soft_mask,
                                       config.mask_width, config.preserve_dc)
                del text_logits
                for qs in range(0, images, chunks.query_chunk_size):
                    qe = min(qs + chunks.query_chunk_size, images)
                    scores = (qi[:, :, qs:qe] @ kh.transpose(-2, -1)) * scale
                    scores[:, :, :, images:] = edited[:, :, qs:qe]
                    output = scores.softmax(-1) @ vh
                    if not torch.isfinite(output).all():
                        raise FloatingPointError("Nonfinite edited attention output")
                    result[row:row+1, start:end, qs:qe] = output.to(v.dtype)
                    del scores, output
                del qi, kh, vh, edited
    return result


def replace_rows(baseline, image_rows, selected, layout, heads, skip_output_reshape=False):
    if not torch.isfinite(image_rows).all():
        raise FloatingPointError("Nonfinite replacement")
    if image_rows.shape[:3] != (len(selected), heads, layout.image_tokens):
        raise ValueError("Invalid edited image-row shape")
    dim = image_rows.shape[-1]
    expected = ((baseline.shape[0], heads, layout.total_tokens, dim) if skip_output_reshape
                else (baseline.shape[0], layout.total_tokens, heads * dim))
    if tuple(baseline.shape) != expected:
        raise ValueError("Native attention output layout changed")
    candidate = baseline.clone()
    for index, row in enumerate(selected):
        if skip_output_reshape:
            candidate[row, :, :layout.image_tokens] = image_rows[index]
        else:
            candidate[row, :layout.image_tokens] = image_rows[index].transpose(0, 1).reshape(
                layout.image_tokens, heads * dim)
    return candidate
