"""Conditional U-Net noise predictor (DDPM-style).

Standard architecture (Ho et al. 2020 / Nichol & Dhariwal): timestep-conditioned
residual blocks, self-attention at the deepest levels, skip connections. The
conditioning image is channel-concatenated to the noisy input before ``forward``,
so ``in_channels = target_channels + cond_channels``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_multiple(x: torch.Tensor, m: int):
    """Replicate-pad H,W up to a multiple of ``m`` (the U-Net needs sizes
    divisible by 2**(levels-1)). Returns padded tensor and original (h, w)."""
    h, w = x.shape[-2:]
    H = (h + m - 1) // m * m
    W = (w + m - 1) // m * m
    return F.pad(x, (0, W - w, 0, H - h), mode="replicate"), (h, w)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _norm(channels: int) -> nn.GroupNorm:
    groups = 32
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(t_dim, out_ch)
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(t))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = _norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        attn = torch.softmax(q @ k / math.sqrt(c), dim=-1)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class Block(nn.Module):
    """A residual block optionally followed by self-attention."""

    def __init__(self, in_ch, out_ch, t_dim, attn, dropout):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, t_dim, dropout)
        self.attn = Attention(out_ch) if attn else nn.Identity()

    def forward(self, x, t):
        return self.attn(self.res(x, t))


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_levels=(2, 3),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_channels = base_channels
        t_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- encoder ----
        ch = base_channels
        chs = [ch]
        self.downs = nn.ModuleList()
        n_levels = len(channel_mults)
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(Block(ch, out_ch, t_dim, level in attn_levels, dropout))
                ch = out_ch
                chs.append(ch)
            if level != n_levels - 1:
                self.downs.append(Downsample(ch))
                chs.append(ch)

        # ---- bottleneck ----
        self.mid1 = ResBlock(ch, ch, t_dim, dropout)
        self.mid_attn = Attention(ch)
        self.mid2 = ResBlock(ch, ch, t_dim, dropout)

        # ---- decoder ----
        self.ups = nn.ModuleList()
        for level in reversed(range(n_levels)):
            out_ch = base_channels * channel_mults[level]
            for _ in range(num_res_blocks + 1):
                self.ups.append(
                    Block(ch + chs.pop(), out_ch, t_dim, level in attn_levels, dropout)
                )
                ch = out_ch
            if level != 0:
                self.ups.append(Upsample(ch))

        self.out_norm = _norm(ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(timestep_embedding(t, self.base_channels))
        h = self.in_conv(x)
        hs = [h]
        for module in self.downs:
            h = module(h, t) if isinstance(module, Block) else module(h)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, t)), t)
        for module in self.ups:
            if isinstance(module, Block):
                h = module(torch.cat([h, hs.pop()], dim=1), t)
            else:
                h = module(h)
        return self.out_conv(F.silu(self.out_norm(h)))
