"""Helpers shared across metrics."""

import torch


def to_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure a 4D [N, C, H, W] tensor."""
    return x.unsqueeze(0) if x.dim() == 3 else x


def crop_border(x: torch.Tensor, border: int) -> torch.Tensor:
    """Drop ``border`` pixels on each side (standard SR eval practice; usually
    equal to the scale factor)."""
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma channel, on [0, 1] RGB tensors [N, C, H, W] -> [N, 1, H, W].
    PSNR/SSIM in SR papers are commonly reported on the Y channel."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 16 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
