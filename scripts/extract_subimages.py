"""Cut DF2K HR images into 480x480 tiles (BasicSR-style sub-images).

Decoding a 2K PNG to crop one small training patch is the loader bottleneck;
pre-cut tiles decode ~10x faster. Non-overlapping grid, edge remainders dropped.

    python scripts/extract_subimages.py   # -> data/DF2K_sub/
"""

import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

SRC_DIRS = ["data/DIV2K/DIV2K_train_HR", "data/Flickr2K/Flickr2K_HR"]
OUT = Path("data/DF2K_sub")
TILE = 480


def process(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    n = 0
    for top in range(0, h - TILE + 1, TILE):
        for left in range(0, w - TILE + 1, TILE):
            tile = img.crop((left, top, left + TILE, top + TILE))
            tile.save(OUT / f"{path.stem}_{top}_{left}.png", compress_level=1)
            n += 1
    return n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for d in SRC_DIRS:
        p = Path(d)
        if p.is_dir():
            files += sorted(p.glob("*.png"))
    print(f"{len(files)} source images -> {OUT}")
    with Pool(8) as pool:
        counts = pool.map(process, files)
    print(f"done: {sum(counts)} tiles")


if __name__ == "__main__":
    main()
