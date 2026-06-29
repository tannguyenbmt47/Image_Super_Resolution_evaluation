"""Gaussian diffusion process (DDPM forward + DDIM sampling).

Model-agnostic: the wrapper (SR3) supplies a ``denoise_fn(x_t, t, cond)`` that
predicts the noise, and provides ``x0`` in the space it diffuses (SR3: HR in
[-1, 1]). Sampling uses DDIM so evaluation needs ~``sampling_timesteps`` network
calls instead of all ``timesteps``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(schedule: str, n: int) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 2e-2, n)
    if schedule == "cosine":  # Nichol & Dhariwal 2021
        steps = n + 1
        x = torch.linspace(0, n, steps)
        ac = torch.cos(((x / n) + 0.008) / 1.008 * math.pi / 2) ** 2
        ac = ac / ac[0]
        betas = 1 - ac[1:] / ac[:-1]
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown beta schedule '{schedule}'")


def _extract(a: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    return a.gather(0, t).reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        loss_type: str = "l1",
        sampling_timesteps: int = 100,
        ddim_eta: float = 0.0,
        clip_denoised: bool = True,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.timesteps = timesteps
        self.sampling_timesteps = min(sampling_timesteps, timesteps)
        self.ddim_eta = ddim_eta
        self.loss_type = loss_type
        self.clip_denoised = clip_denoised

        betas = make_beta_schedule(beta_schedule, timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_acp", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_acp", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x0, t, noise):
        return (_extract(self.sqrt_acp, t, x0.shape) * x0
                + _extract(self.sqrt_one_minus_acp, t, x0.shape) * noise)

    def p_losses(self, x0, cond):
        b = x0.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.denoise_fn(x_t, t, cond)
        loss = F.l1_loss if self.loss_type == "l1" else F.mse_loss
        return loss(pred, noise)

    @torch.no_grad()
    def sample(self, shape, cond):
        device = self.alphas_cumprod.device
        acp = self.alphas_cumprod
        times = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        times = times.round().long().flip(0).tolist()  # [T-1, ..., -1]
        pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        for t, t_next in pairs:
            t_b = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = self.denoise_fn(x, t_b, cond)
            a_t = acp[t]
            x0 = (x - (1 - a_t).sqrt() * pred_noise) / a_t.sqrt()
            if self.clip_denoised:
                x0 = x0.clamp(-1, 1)
            if t_next < 0:
                x = x0
                break
            a_next = acp[t_next]
            sigma = self.ddim_eta * (
                ((1 - a_t / a_next) * (1 - a_next) / (1 - a_t)).clamp(min=0)
            ).sqrt()
            c = (1 - a_next - sigma ** 2).clamp(min=0).sqrt()
            x = a_next.sqrt() * x0 + c * pred_noise + sigma * torch.randn_like(x)
        return x
