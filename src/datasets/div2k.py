"""DIV2K / DF2K training dataset.

Expects one or more folders of HR images. Pass a single path for plain DIV2K, or
a list of paths for DF2K (DIV2K train + Flickr2K). LR images are synthesized on
the fly (bicubic or realistic degradation), so only HR files are needed. Missing
folders in the list are skipped (so a DF2K config still works with only DIV2K
present) -- an error is raised only if no images are found at all.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from . import DATASETS
from . import transforms as T

_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@DATASETS.register("DIV2K")
class DIV2K(Dataset):
    def __init__(
        self,
        hr_dir,                       # str path or list of paths (DF2K)
        scale: int = 4,
        patch_size: int = 192,
        pre_upscale: bool = False,
        degradation: str = "bicubic",
        augment: bool = True,
        repeat: int = 1,
    ):
        self.scale = scale
        self.patch_size = patch_size
        self.pre_upscale = pre_upscale
        self.degradation = degradation
        self.do_augment = augment
        self.repeat = repeat

        dirs = [hr_dir] if isinstance(hr_dir, str) else list(hr_dir)
        self.files = []
        for d in dirs:
            p = Path(d)
            if p.is_dir():
                self.files += [f for f in p.iterdir() if f.suffix.lower() in _EXTS]
        self.files = sorted(self.files)
        if not self.files:
            raise FileNotFoundError(f"No images found in {dirs}")

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        path = self.files[idx % len(self.files)]
        hr = T.to_tensor(Image.open(path))
        hr = T.paired_random_crop(hr, self.scale, self.patch_size)
        if self.do_augment:
            hr = T.augment(hr)
        lr = T.make_lr(hr, self.scale, self.pre_upscale, self.degradation)
        return {"lr": lr, "hr": hr, "name": path.stem}
