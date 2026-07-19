"""Benchmark test datasets: Set5, Set14, BSD100, Urban100, Manga109.

These share one loader -- they are all just a folder of HR images. Point
``hr_dir`` at the benchmark's HR folder; LR is synthesized by bicubic
downscaling so reported PSNR/SSIM match the standard bicubic SR protocol.

Full images are returned (no cropping), with HR sizes made divisible by the
scale factor so LR/SR/HR line up exactly.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from . import DATASETS
from . import transforms as T

_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@DATASETS.register("Benchmark")
class Benchmark(Dataset):
    """Folder of HR images with synthesized LR. Used for Set5/.../Manga109 and
    also DIV2K-Val (point ``hr_dir`` at DIV2K_valid_HR)."""

    def __init__(self, hr_dir: str, scale: int = 4, pre_upscale: bool = False,
                 degradation: str = "bicubic", max_hr: int = 0, limit: int = 0,
                 name: str = "benchmark"):
        self.hr_dir = Path(hr_dir)
        self.scale = scale
        self.pre_upscale = pre_upscale
        self.degradation = degradation
        # >0: center-crop HR to at most max_hr x max_hr. Needed for diffusion eval,
        # whose UNet self-attention is infeasible on full 2K DIV2K-Val images.
        self.max_hr = max_hr
        self.name = name
        self.files = sorted(p for p in self.hr_dir.iterdir() if p.suffix.lower() in _EXTS)
        if limit > 0:  # first-N subset (cheap in-training validation for diffusion)
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No images found in {self.hr_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        hr = T.to_tensor(Image.open(path))
        _, h, w = hr.shape
        if self.max_hr > 0:  # center crop to a manageable size
            ch, cw = min(h, self.max_hr), min(w, self.max_hr)
            top, left = (h - ch) // 2, (w - cw) // 2
            hr = hr[:, top:top + ch, left:left + cw]
            _, h, w = hr.shape
        # crop HR so each side is divisible by scale
        hr = hr[:, : h - h % self.scale, : w - w % self.scale]
        lr = T.make_lr(hr, self.scale, self.pre_upscale, self.degradation)
        return {"lr": lr, "hr": hr, "name": path.stem}
