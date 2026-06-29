"""Peak Signal-to-Noise Ratio (higher is better)."""

import torch

from . import METRICS
from .common import crop_border, rgb_to_y, to_batch


@METRICS.register("psnr")
class PSNR:
    def __init__(self, crop: int = 4, y_channel: bool = True):
        self.crop = crop
        self.y_channel = y_channel

    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> float:
        sr, hr = to_batch(sr).clamp(0, 1), to_batch(hr).clamp(0, 1)
        if self.y_channel:
            sr, hr = rgb_to_y(sr), rgb_to_y(hr)
        sr, hr = crop_border(sr, self.crop), crop_border(hr, self.crop)
        mse = torch.mean((sr - hr) ** 2, dim=[1, 2, 3])
        psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
        return psnr.mean().item()
