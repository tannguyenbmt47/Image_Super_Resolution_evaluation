import pytest

from src.datasets import build_dataset
from src.utils import Config


@pytest.mark.integration
def test_div2k_patches(hr_dir):
    ds = build_dataset(Config({"name": "DIV2K", "args": {
        "hr_dir": hr_dir, "scale": 4, "patch_size": 16, "repeat": 2}}))
    assert len(ds) == 8  # 4 images * repeat 2
    s = ds[0]
    assert s["lr"].shape == (3, 4, 4)
    assert s["hr"].shape == (3, 16, 16)
    assert s["lr"].min() >= 0 and s["lr"].max() <= 1


@pytest.mark.integration
def test_benchmark_full_images(hr_dir):
    ds = build_dataset(Config({"name": "Benchmark", "args": {
        "hr_dir": hr_dir, "scale": 4, "name": "set"}}))
    assert len(ds) == 4
    s = ds[0]
    assert s["hr"].shape[-1] % 4 == 0
    assert s["lr"].shape[-1] * 4 == s["hr"].shape[-1]


@pytest.mark.integration
def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_dataset(Config({"name": "Benchmark", "args": {
            "hr_dir": str(tmp_path / "does_not_exist"), "scale": 4}}))
