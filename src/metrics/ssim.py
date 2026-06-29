"""Structural Similarity Index (higher is better).

Gaussian-window SSIM (Wang et al., 2004) implemented in torch so it runs on GPU
batches. Defaults match the common SR protocol: Y channel, 11x11 window.
"""

import torch
import torch.nn.functional as F

from . import METRICS
from .common import crop_border, rgb_to_y, to_batch


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


@METRICS.register("ssim")
class SSIM:
    def __init__(self, crop: int = 4, y_channel: bool = True,
                 window_size: int = 11, sigma: float = 1.5):
        self.crop = crop
        self.y_channel = y_channel
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> float:
        sr, hr = to_batch(sr).clamp(0, 1), to_batch(hr).clamp(0, 1)
        if self.y_channel:
            sr, hr = rgb_to_y(sr), rgb_to_y(hr)
        sr, hr = crop_border(sr, self.crop), crop_border(hr, self.crop)

        channels = sr.shape[1]
        window = _gaussian_window(self.window_size, self.sigma, channels).to(sr)
        pad = self.window_size // 2

        def filt(x):
            return F.conv2d(x, window, padding=pad, groups=channels)

        mu1, mu2 = filt(sr), filt(hr)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = filt(sr * sr) - mu1_sq
        sigma2_sq = filt(hr * hr) - mu2_sq
        sigma12 = filt(sr * hr) - mu12

        ssim_map = ((2 * mu12 + self.C1) * (2 * sigma12 + self.C2)) / (
            (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        )
        return ssim_map.mean().item()
