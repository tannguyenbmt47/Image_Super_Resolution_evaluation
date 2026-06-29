"""Pixel and perceptual losses for feed-forward SR training.

``build_loss`` turns a training config into a single criterion module:
  - ``loss``: pixel term -- ``l1`` | ``l2`` | ``charbonnier``
  - ``perceptual_weight``: if > 0, add ``w * VGG(relu2_2)`` perceptual L1

Charbonnier is a smooth L1 variant common in SR. Perceptual loss
trades PSNR/SSIM for sharper, more realistic texture (as the plan notes).
Diffusion models do not use this -- they define their own noise-prediction loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, x, y):
        return torch.sqrt((x - y) ** 2 + self.eps2).mean()


class VGGPerceptualLoss(nn.Module):
    """L1 distance in VGG16 relu2_2 feature space (inputs in [0, 1] RGB)."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        self.slice = nn.Sequential(*[vgg[i] for i in range(9)])  # up to relu2_2
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return F.l1_loss(self.slice(x), self.slice(y))


_PIXEL = {"l1": nn.L1Loss, "l2": nn.MSELoss, "charbonnier": CharbonnierLoss}


class CombinedLoss(nn.Module):
    def __init__(self, pixel: str = "l1", perceptual_weight: float = 0.0):
        super().__init__()
        if pixel not in _PIXEL:
            raise ValueError(f"unknown pixel loss '{pixel}' (have {list(_PIXEL)})")
        self.pixel = _PIXEL[pixel]()
        self.perceptual_weight = perceptual_weight
        self.perceptual = VGGPerceptualLoss() if perceptual_weight > 0 else None

    def forward(self, sr, hr):
        loss = self.pixel(sr, hr)
        if self.perceptual is not None:
            loss = loss + self.perceptual_weight * self.perceptual(sr, hr)
        return loss


def build_loss(cfg) -> CombinedLoss:
    return CombinedLoss(
        pixel=cfg.get("loss", "l1"),
        perceptual_weight=cfg.get("perceptual_weight", 0.0),
    )
