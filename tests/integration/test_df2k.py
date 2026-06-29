import numpy as np
import pytest
from PIL import Image

from src.datasets import build_dataset
from src.utils import Config


def _extra_dir(tmp_path, n):
    d = tmp_path / "HR2"
    d.mkdir()
    rng = np.random.default_rng(1)
    for i in range(n):
        Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8")).save(d / f"x{i}.png")
    return str(d)


@pytest.mark.integration
def test_div2k_combines_multiple_dirs(hr_dir, tmp_path):
    d2 = _extra_dir(tmp_path, 2)
    ds = build_dataset(Config({"name": "DIV2K", "args": {
        "hr_dir": [hr_dir, d2], "scale": 4, "patch_size": 16, "repeat": 1}}))
    assert len(ds) == 6  # 4 (hr_dir) + 2 (d2)


@pytest.mark.integration
def test_div2k_skips_missing_dir(hr_dir, tmp_path):
    # a DF2K config still works when Flickr2K (a listed dir) is absent
    ds = build_dataset(Config({"name": "DIV2K", "args": {
        "hr_dir": [hr_dir, str(tmp_path / "absent")], "scale": 4, "patch_size": 16}}))
    assert len(ds) == 4


@pytest.mark.integration
def test_div2k_realistic_degradation(hr_dir):
    ds = build_dataset(Config({"name": "DIV2K", "args": {
        "hr_dir": hr_dir, "scale": 4, "patch_size": 16, "degradation": "realistic"}}))
    s = ds[0]
    assert s["lr"].shape == (3, 4, 4)
    assert s["hr"].shape == (3, 16, 16)
