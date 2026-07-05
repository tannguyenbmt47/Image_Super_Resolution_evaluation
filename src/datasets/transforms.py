"""Paired LR/HR transforms shared by all datasets.

Convention: HR is the ground-truth image; LR is synthesized from HR. Two
degradation modes are supported (``make_lr(..., degradation=...)``):
  - ``"bicubic"``  : classical SR -- a single bicubic downscale. Clean and the
    standard setting for PSNR/SSIM/LPIPS comparison.
  - ``"realistic"``: real-world SR -- random blur + downsample + noise + JPEG, a
    first-order Real-ESRGAN/BSRGAN-style pipeline for robustness to real images.
For SRCNN-style models the LR is then bicubic-upscaled back to HR size
(``pre_upscale``).
"""

import io
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> float tensor [C, H, W] in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resize(x: torch.Tensor, size, mode: str) -> torch.Tensor:
    kwargs = {} if mode == "area" else {"align_corners": False}
    if mode in ("bicubic", "bilinear") and size[0] < x.shape[-2]:
        # Downscaling must antialias to match the standard SR protocol
        # (MATLAB/PIL imresize). F.interpolate aliases by default, which is
        # out-of-distribution for models trained on standard bicubic LR and
        # makes reported metrics incomparable with published results.
        kwargs["antialias"] = True
    return F.interpolate(x, size=size, mode=mode, **kwargs)


def _jpeg(img: torch.Tensor, quality: int) -> torch.Tensor:
    """Round-trip an image through JPEG compression at the given quality."""
    arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(out).permute(2, 0, 1)


def degrade_realistic(hr: torch.Tensor, scale: int) -> torch.Tensor:
    """First-order real-world degradation: blur -> downsample -> noise -> JPEG."""
    _, h, w = hr.shape
    img = hr
    if random.random() < 0.8:  # blur
        k = random.choice([3, 5, 7])
        img = gaussian_blur(img, kernel_size=k, sigma=random.uniform(0.2, 2.0))
    mode = random.choice(["bicubic", "bilinear", "area"])  # downsample
    img = _resize(img.unsqueeze(0), (h // scale, w // scale), mode).squeeze(0).clamp(0, 1)
    if random.random() < 0.5:  # gaussian noise
        img = (img + torch.randn_like(img) * (random.uniform(1, 15) / 255.0)).clamp(0, 1)
    if random.random() < 0.7:  # JPEG artifacts
        img = _jpeg(img, random.randint(40, 95))
    return img.clamp(0, 1)


def make_lr(hr: torch.Tensor, scale: int, pre_upscale: bool = False,
            degradation: str = "bicubic") -> torch.Tensor:
    """Synthesize LR from HR by ``degradation``; optionally upscale to HR size."""
    _, h, w = hr.shape
    if degradation == "bicubic":
        lr = _resize(hr.unsqueeze(0), (h // scale, w // scale), "bicubic").squeeze(0).clamp(0, 1)
    elif degradation == "realistic":
        lr = degrade_realistic(hr, scale)
    else:
        raise ValueError(f"unknown degradation '{degradation}'")
    if pre_upscale:
        lr = _resize(lr.unsqueeze(0), (h, w), "bicubic").squeeze(0).clamp(0, 1)
    return lr


def paired_random_crop(hr: torch.Tensor, scale: int, patch: int) -> torch.Tensor:
    """Crop a random ``patch``x``patch`` HR region (aligned to ``scale``)."""
    _, h, w = hr.shape
    patch = (patch // scale) * scale
    top = random.randint(0, h - patch)
    left = random.randint(0, w - patch)
    return hr[:, top:top + patch, left:left + patch]


def augment(hr: torch.Tensor) -> torch.Tensor:
    """Random horizontal/vertical flip + 90-degree rotation."""
    if random.random() < 0.5:
        hr = torch.flip(hr, dims=[2])
    if random.random() < 0.5:
        hr = torch.flip(hr, dims=[1])
    if random.random() < 0.5:
        hr = torch.rot90(hr, k=1, dims=[1, 2])
    return hr
