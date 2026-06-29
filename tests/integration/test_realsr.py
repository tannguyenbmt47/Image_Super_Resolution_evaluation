import numpy as np
import pytest
from PIL import Image

from src.datasets import build_dataset
from src.utils import Config


@pytest.fixture
def realsr_dirs(tmp_path):
    lr, hr = tmp_path / "LR", tmp_path / "HR"
    lr.mkdir()
    hr.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        Image.fromarray((rng.random((16, 16, 3)) * 255).astype("uint8")).save(lr / f"{i}.png")
        Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).save(hr / f"{i}.png")
    return str(lr), str(hr)


@pytest.mark.integration
def test_realsr_full_image(realsr_dirs):
    lr, hr = realsr_dirs
    ds = build_dataset(Config({"name": "RealSR", "args": {"lr_dir": lr, "hr_dir": hr, "scale": 4}}))
    assert len(ds) == 3
    s = ds[0]
    assert s["hr"].shape == (3, 64, 64)
    assert s["lr"].shape == (3, 16, 16)


@pytest.mark.integration
def test_realsr_aligned_patch(realsr_dirs):
    lr, hr = realsr_dirs
    ds = build_dataset(Config({"name": "RealSR", "args": {
        "lr_dir": lr, "hr_dir": hr, "scale": 4, "patch_size": 32}}))
    s = ds[0]
    assert s["lr"].shape == (3, 8, 8)   # 32 / scale
    assert s["hr"].shape == (3, 32, 32)


@pytest.mark.integration
def test_realsr_count_mismatch_raises(tmp_path):
    lr, hr = tmp_path / "LR", tmp_path / "HR"
    lr.mkdir()
    hr.mkdir()
    Image.fromarray(np.zeros((16, 16, 3), "uint8")).save(lr / "a.png")
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(hr / "a.png")
    Image.fromarray(np.zeros((64, 64, 3), "uint8")).save(hr / "b.png")
    with pytest.raises(ValueError):
        build_dataset(Config({"name": "RealSR", "args": {
            "lr_dir": str(lr), "hr_dir": str(hr), "scale": 4}}))
