"""SRGAN (Ledig et al., 2017) -- Photo-Realistic Single Image Super-Resolution
Using a Generative Adversarial Network.

Implements the unified interface used by the engine:
  - ``compute_loss(lr, hr)`` returns ``{"g_loss": Tensor, "d_loss": Tensor}``
    (the Trainer detects the dict return and runs adversarial training).
  - ``forward(lr)`` / ``super_resolve(lr)`` produce SR from the Generator only.

Architecture follows the original paper:
  - **Generator (SRResNet)**: 16 residual blocks + 2× PixelShuffle upsampling
    (total 4× for x4 scale). Takes raw LR, outputs HR-sized SR. Uses PReLU
    and BatchNorm. ``pre_upscale`` must be ``false``.
  - **Discriminator**: 8 conv blocks with increasing channels + global avg pool
    + linear → scalar. PatchGAN-style.

Losses:
  - Pixel: MSE (default, per paper) or configurable
  - Perceptual: VGG54 (relu5_4 features of VGG19, deeper than the VGG16 relu2_2
    used by the default CombinedLoss)
  - Adversarial: vanilla GAN (-log D(G(lr)))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses import VGG54PerceptualLoss
from . import MODELS

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Residual block: Conv-BN-PReLU-Conv-BN + skip connection."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """2× spatial upsampling via PixelShuffle (sub-pixel convolution)."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(channels),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    """Discriminator conv block: Conv-BN-LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# Generator (SRResNet)
# ---------------------------------------------------------------------------


class Generator(nn.Module):
    """SRResNet generator: 16 residual blocks + 2× PixelShuffle upsampling.

    Input: raw LR image [N, C, H/scale, W/scale].
    Output: SR image [N, C, H, W] (4× spatial upsampling via PixelShuffle).

    Unlike SRCNN, the Generator has internal upsampling, so ``pre_upscale``
    must be ``false`` in dataset configs.
    """

    def __init__(
        self, num_channels: int = 3, num_features: int = 64, num_blocks: int = 16
    ):
        super().__init__()
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 9, padding=4, bias=False),
            nn.PReLU(num_features),
        )
        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_blocks)]
        )
        # Post-residual convolution (with skip from initial)
        self.post_residual = nn.Sequential(
            nn.Conv2d(num_features, num_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
        )
        # 2× upsample blocks (total 4× for x4 scale)
        self.upsample = nn.Sequential(
            UpsampleBlock(num_features),
            UpsampleBlock(num_features),
        )
        # Final reconstruction
        self.final = nn.Sequential(
            nn.Conv2d(num_features, num_channels, 9, padding=4, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        initial = self.initial(x)
        residual = self.residuals(initial)
        post = self.post_residual(residual) + initial
        up = self.upsample(post)
        return self.final(up)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------


class Discriminator(nn.Module):
    """PatchGAN discriminator for SRGAN.

    Takes an HR-sized image (real or SR) and outputs a scalar per image
    (after global average pooling).
    """

    def __init__(self, num_channels: int = 3, num_features: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, num_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(num_features, num_features, stride=2),
            ConvBlock(num_features, num_features * 2),
            ConvBlock(num_features * 2, num_features * 2, stride=2),
            ConvBlock(num_features * 2, num_features * 4),
            ConvBlock(num_features * 4, num_features * 4, stride=2),
            ConvBlock(num_features * 4, num_features * 8),
            ConvBlock(num_features * 8, num_features * 8, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ---------------------------------------------------------------------------
# SRGAN model (wraps Generator + Discriminator)
# ---------------------------------------------------------------------------


@MODELS.register("SRGAN")
class SRGAN(nn.Module):
    """SRGAN: adversarial training for photo-realistic SR.

    The engine dispatches on the return type of ``compute_loss``:
    - dict → adversarial training (``{"g_loss", "d_loss"}``)
    - scalar → standard training (feed-forward model)

    For inference, only the Generator is used (``forward`` / ``super_resolve``).
    """

    def __init__(
        self,
        num_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        scale: int = 4,
        perceptual_layer: str = "relu5_4",
        perceptual_weight: float = 1e-3,
        adversarial_weight: float = 1e-3,
        loss: str = "mse",
    ):
        super().__init__()
        self.scale = scale
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.generator = Generator(num_channels, num_features, num_blocks)
        self.discriminator = Discriminator(num_channels, num_features)
        # Lazy-loaded perceptual loss (VGG54 or VGG19 relu5_4)
        self._perceptual = None
        self._perceptual_layer = perceptual_layer
        # Pixel loss
        _pixel_losses = {"mse": nn.MSELoss, "l2": nn.MSELoss, "l1": nn.L1Loss}
        if loss not in _pixel_losses:
            raise ValueError(
                f"unknown pixel loss '{loss}' (have {list(_pixel_losses)})"
            )
        self._pixel_loss = _pixel_losses[loss]()

    def _get_perceptual(self, device):
        if self._perceptual is None:
            self._perceptual = VGG54PerceptualLoss().to(device)
        return self._perceptual

    def compute_loss(self, lr, hr):
        """Return ``{"g_loss": Tensor, "d_loss": Tensor}`` for GAN training.

        Generator loss = pixel + perceptual + adversarial
        Discriminator loss = BCE(real→1, fake→0)
        """
        sr = self.generator(lr)

        # --- Generator loss ---
        g_pixel = self._pixel_loss(sr, hr)
        g_perceptual = self._get_perceptual(sr.device)(sr, hr)
        g_adversarial = -torch.log(
            self.discriminator(sr.clamp(0, 1)).sigmoid() + 1e-8
        ).mean()
        g_loss = (
            g_pixel
            + self.perceptual_weight * g_perceptual
            + self.adversarial_weight * g_adversarial
        )

        # --- Discriminator loss ---
        d_real = self.discriminator(hr)
        d_fake = self.discriminator(sr.detach().clamp(0, 1))
        d_loss = F.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real)
        ) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

        return {"g_loss": g_loss, "d_loss": d_loss}

    def forward(self, x):
        return self.generator(x)

    @torch.no_grad()
    def super_resolve(self, lr):
        return self.generator(lr).clamp(0, 1)
