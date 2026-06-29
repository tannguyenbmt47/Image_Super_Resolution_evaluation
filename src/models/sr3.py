"""SR3 -- Image Super-Resolution via Iterative Refinement (Saharia et al., 2021).

A conditional DDPM that diffuses the **HR image** (in [-1, 1]). The condition is
the LR image bicubic-upsampled to HR size and channel-concatenated to the noisy
input, so the U-Net sees ``[cond(3) ; x_t(3)] -> noise(3)``.

Implements the unified training/inference interface used by the engine:
``compute_loss(lr, hr)`` (diffusion loss) and ``super_resolve(lr)`` (sampling).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import MODELS
from .diffusion import GaussianDiffusion, UNet, pad_to_multiple


@MODELS.register("SR3")
class SR3(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        num_channels: int = 3,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_levels=(2, 3),
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        loss_type: str = "l1",
        sampling_timesteps: int = 100,
        ddim_eta: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.scale = scale
        self.num_channels = num_channels
        self.window = 2 ** (len(channel_mults) - 1)
        self.unet = UNet(
            in_channels=num_channels * 2,
            out_channels=num_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_res_blocks=num_res_blocks,
            attn_levels=attn_levels,
            dropout=dropout,
        )
        self.diffusion = GaussianDiffusion(
            self._denoise, timesteps=timesteps, beta_schedule=beta_schedule,
            loss_type=loss_type, sampling_timesteps=sampling_timesteps, ddim_eta=ddim_eta,
        )

    def _denoise(self, x_t, t, cond):
        return self.unet(torch.cat([cond, x_t], dim=1), t)

    def _condition(self, lr, size):
        up = F.interpolate(lr, size=size, mode="bicubic", align_corners=False).clamp(0, 1)
        return up * 2 - 1  # to [-1, 1]

    def compute_loss(self, lr, hr):
        cond = self._condition(lr, hr.shape[-2:])
        return self.diffusion.p_losses(hr * 2 - 1, cond)

    @torch.no_grad()
    def super_resolve(self, lr):
        H, W = lr.shape[-2] * self.scale, lr.shape[-1] * self.scale
        cond = self._condition(lr, (H, W))
        cond, _ = pad_to_multiple(cond, self.window)
        shape = (lr.shape[0], self.num_channels, cond.shape[-2], cond.shape[-1])
        x0 = self.diffusion.sample(shape, cond)
        sr = ((x0 + 1) / 2).clamp(0, 1)
        return sr[..., :H, :W]
