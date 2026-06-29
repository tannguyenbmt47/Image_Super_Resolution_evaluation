"""RealSR-style paired real-world dataset.

Unlike the synthetic datasets, here LR is a *real* captured low-resolution image,
not synthesized from HR -- this is what the fine-tuning stage (stage 2) uses and
what the RealSR evaluation reports on. Provide two folders with matching filenames:

    lr_dir/  img001.png ...
    hr_dir/  img001.png ...

Pairs are matched by sorted order. HR is cropped so each side is divisible by
``scale``; LR is then center-cropped to ``HR/scale`` so the pair lines up exactly
(captured pairs are registered but can be off by a pixel or two).
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from . import DATASETS
from . import transforms as T

_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def _list(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in _EXTS)


@DATASETS.register("RealSR")
class RealSR(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str, scale: int = 4,
                 patch_size: int = 0, name: str = "RealSR"):
        self.scale = scale
        self.patch_size = patch_size  # >0 -> random training patches; 0 -> full image
        self.name = name
        self.lr_files = _list(lr_dir)
        self.hr_files = _list(hr_dir)
        if not self.lr_files or not self.hr_files:
            raise FileNotFoundError(f"Missing images in {lr_dir} / {hr_dir}")
        if len(self.lr_files) != len(self.hr_files):
            raise ValueError(
                f"LR/HR count mismatch: {len(self.lr_files)} vs {len(self.hr_files)}")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        import random

        lr = T.to_tensor(Image.open(self.lr_files[idx]))
        hr = T.to_tensor(Image.open(self.hr_files[idx]))
        s = self.scale

        # clamp HR to a multiple of scale and LR to the matching HR/scale extent
        _, h, w = hr.shape
        h, w = min(h - h % s, lr.shape[1] * s), min(w - w % s, lr.shape[2] * s)
        hr = hr[:, :h, :w]
        lr = lr[:, : h // s, : w // s]

        if self.patch_size > 0:  # training: aligned random patch on the LR grid
            p = self.patch_size // s
            top = random.randint(0, lr.shape[1] - p)
            left = random.randint(0, lr.shape[2] - p)
            lr = lr[:, top:top + p, left:left + p]
            hr = hr[:, top * s:(top + p) * s, left * s:(left + p) * s]
        return {"lr": lr, "hr": hr, "name": self.hr_files[idx].stem}
