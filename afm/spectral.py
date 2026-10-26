import math

import torch


def schedule_alphas(strength: float, progress: float) -> tuple[float, float]:
    if not math.isfinite(strength) or not math.isfinite(progress) or not 0 <= progress <= 1:
        raise ValueError("strength must be finite and progress must be in [0, 1]")
    return 1 + strength * (1 - progress), 1 + strength * progress


def filter_logits(logits, grid, alpha_lf, alpha_hf, cutoff=0.25,
                  soft_mask=True, mask_width=0.05, preserve_dc=True):
    h, w = grid
    if (type(h) is not int or type(w) is not int or min(h, w) <= 0
            or logits.ndim != 4 or logits.shape[2] != h * w):
        raise ValueError("Expected [B,heads,H*W,T] logits and actual positive grid")
    if not 0 < cutoff < 1 or not 0 < mask_width <= 1:
        raise ValueError("Invalid frequency cutoff or mask width")
    if not all(math.isfinite(x) for x in (alpha_lf, alpha_hf)):
        raise ValueError("Frequency gains must be finite")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        signal = logits.float()
        if not torch.isfinite(signal).all():
            raise FloatingPointError("Nonfinite AFM logits")
        fy = torch.fft.fftfreq(h, device=signal.device, dtype=torch.float32)[:, None]
        fx = torch.fft.fftfreq(w, device=signal.device, dtype=torch.float32)[None, :]
        radius = torch.sqrt(fy.square() + fx.square()) / math.sqrt(0.5)
        if soft_mask:
            transition = ((radius - cutoff + mask_width / 2) / mask_width).clamp(0, 1)
            low = 0.5 * (1 + torch.cos(torch.pi * transition))
        else:
            low = (radius <= cutoff).float()
        gain = alpha_lf * low + alpha_hf * (1 - low)
        if preserve_dc:
            gain[0, 0] = 1
        maps = signal.reshape(*signal.shape[:2], h, w, signal.shape[-1]).permute(0, 1, 4, 2, 3)
        spectrum = torch.fft.fft2(maps, dim=(-2, -1))
        restored = torch.fft.ifft2(spectrum * gain, dim=(-2, -1)).real
        result = restored.permute(0, 1, 3, 4, 2).reshape_as(signal).contiguous()
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite filtered AFM logits")
        return result
