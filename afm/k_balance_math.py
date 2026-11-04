from dataclasses import dataclass
import math

import torch


STRENGTH_LIMIT = 10.0
MAX_CHANGE_LIMIT = 0.95


@dataclass(frozen=True)
class KBalanceConfig:
    strength: float = 0.0
    max_change: float = 0.03

    def __post_init__(self):
        for name in ("strength", "max_change"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not -STRENGTH_LIMIT <= self.strength <= STRENGTH_LIMIT:
            raise ValueError(f"strength must be in [-{STRENGTH_LIMIT:g}, {STRENGTH_LIMIT:g}]")
        if not 0 <= self.max_change <= MAX_CHANGE_LIMIT:
            raise ValueError(f"max_change must be in [0, {MAX_CHANGE_LIMIT:g}]")

    @property
    def bypass(self) -> bool:
        return self.strength == 0 or self.max_change == 0


def _validate_inputs(
    k_codes: torch.Tensor,
    k_scales: torch.Tensor,
    head_dim: int,
    config: KBalanceConfig,
    chunk_rows: int,
) -> None:
    if not isinstance(config, KBalanceConfig):
        raise ValueError("config must be a KBalanceConfig")
    if not isinstance(k_codes, torch.Tensor) or not isinstance(k_scales, torch.Tensor):
        raise ValueError("k_codes and k_scales must be tensors")
    if k_codes.device.type != "cpu" or k_scales.device.type != "cpu":
        raise ValueError("k_codes and k_scales must be CPU tensors")
    if k_codes.dtype != torch.int8 or k_codes.ndim != 2:
        raise ValueError("k_codes must be signed int8 [K_rows, in_features]")
    if k_scales.dtype != torch.float32 or k_scales.ndim != 2 or k_scales.shape[1] != 1:
        raise ValueError("k_scales must be float32 [K_rows, 1]")
    if k_codes.shape[0] == 0 or k_codes.shape[1] == 0 or k_scales.shape[0] != k_codes.shape[0]:
        raise ValueError("k_codes and k_scales must have matching nonempty rows")
    if type(head_dim) is not int or head_dim <= 0 or k_codes.shape[0] % head_dim:
        raise ValueError("head_dim must be positive and divide K rows")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")


def _row_code_norms(k_codes: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    norms = torch.empty(k_codes.shape[0], dtype=torch.float64)
    for start in range(0, k_codes.shape[0], chunk_rows):
        stop = min(start + chunk_rows, k_codes.shape[0])
        chunk = k_codes[start:stop].to(dtype=torch.float64)
        norms[start:stop] = chunk.square().sum(dim=1).sqrt()
    return norms


def _quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q).item())


def _metrics(norms: torch.Tensor, head_dim: int) -> dict[str, object]:
    heads = norms.reshape(-1, head_dim)
    head_medians = torch.quantile(heads, 0.5, dim=1)
    head_log_stds = heads.log().std(dim=1, unbiased=False)
    per_head_ratios = torch.quantile(heads, 0.99, dim=1) / head_medians
    return {
        "row_ratio": _quantile(norms, 0.99) / _quantile(norms, 0.5),
        "median_head_log_std": _quantile(head_log_stds, 0.5),
        "per_head_median": _quantile(per_head_ratios, 0.5),
        "per_head_max": float(per_head_ratios.max().item()),
    }


def _report(
    before: torch.Tensor,
    after: torch.Tensor,
    gains: torch.Tensor,
    changed_rows: int,
    clipped_rows: int,
    head_dim: int,
    config: KBalanceConfig,
    chunk_rows: int,
    in_features: int,
) -> dict[str, object]:
    before_metrics = _metrics(before, head_dim)
    after_metrics = _metrics(after, head_dim)
    effective_chunk = min(chunk_rows, before.numel())
    workspace_estimate = (
        effective_chunk * in_features * 16
        + effective_chunk * 24
        + before.numel() * 64
    )
    return {
        "rows": before.numel(),
        "heads": before.numel() // head_dim,
        "head_dim": head_dim,
        "strength": float(config.strength),
        "max_change": float(config.max_change),
        "changed_rows": changed_rows,
        "clipped_rows": clipped_rows,
        "clipped_fraction": clipped_rows / before.numel(),
        "gain_min": float(gains.min().item()),
        "gain_max": float(gains.max().item()),
        "row_p99_over_p50": {
            "before": before_metrics["row_ratio"],
            "after": after_metrics["row_ratio"],
        },
        "median_per_head_log_norm_std": {
            "before": before_metrics["median_head_log_std"],
            "after": after_metrics["median_head_log_std"],
        },
        "per_head_p99_over_p50": {
            "median_before": before_metrics["per_head_median"],
            "max_before": before_metrics["per_head_max"],
            "median_after": after_metrics["per_head_median"],
            "max_after": after_metrics["per_head_max"],
        },
        "workspace_estimate_bytes": int(workspace_estimate),
    }


def compute_k_gains(
    k_codes: torch.Tensor,
    k_scales: torch.Tensor,
    head_dim: int,
    config: KBalanceConfig,
    chunk_rows: int = 128,
) -> tuple[torch.Tensor, dict[str, object]]:
    _validate_inputs(k_codes, k_scales, head_dim, config, chunk_rows)
    scales = k_scales.detach()
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise FloatingPointError("K scales must be finite and positive")

    code_norms = _row_code_norms(k_codes, chunk_rows)
    scales64 = scales[:, 0].to(dtype=torch.float64)
    row_norms = code_norms * scales64
    if not torch.isfinite(row_norms).all() or torch.any(row_norms == 0):
        raise FloatingPointError("K row norms must be finite and nonzero")

    if config.bypass:
        gains = torch.ones((k_codes.shape[0], 1), dtype=torch.float32)
        clipped_rows = 0
    else:
        heads = row_norms.reshape(-1, head_dim)
        medians = torch.quantile(heads, 0.5, dim=1).repeat_interleave(head_dim)
        log_gains = config.strength * (medians.log() - row_norms.log())
        lower_bound = math.log1p(-config.max_change)
        upper_bound = math.log1p(config.max_change)
        clipped_rows = int(
            torch.count_nonzero((log_gains < lower_bound) | (log_gains > upper_bound)).item()
        )
        log_gains.clamp_(lower_bound, upper_bound)
        gains = log_gains.exp().to(dtype=torch.float32).reshape(-1, 1)

    new_scales = scales * gains
    if not torch.isfinite(new_scales).all() or torch.any(new_scales == 0):
        raise FloatingPointError("K scale update overflowed or underflowed float32")
    after_norms = code_norms * new_scales[:, 0].to(dtype=torch.float64)
    if not torch.isfinite(after_norms).all() or torch.any(after_norms == 0):
        raise FloatingPointError("Updated K row norms must be finite and nonzero")

    report = _report(
        row_norms,
        after_norms,
        gains,
        int(torch.count_nonzero(new_scales != scales).item()),
        clipped_rows,
        head_dim,
        config,
        chunk_rows,
        k_codes.shape[1],
    )
    return gains, report
