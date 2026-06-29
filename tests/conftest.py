"""Shared fixtures for the test suite."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def hr_dir(tmp_path):
    """A folder of small synthetic HR PNGs (32x32), enough for SR pipelines."""
    d = tmp_path / "HR"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(4):
        arr = (rng.random((32, 32, 3)) * 255).astype("uint8")
        Image.fromarray(arr).save(d / f"img{i}.png")
    return str(d)


@pytest.fixture
def tiny_diffusion_args():
    """Minimal SR3 hyperparameters so diffusion tests run fast on CPU."""
    return dict(
        base_channels=8,
        channel_mults=[1, 2],
        num_res_blocks=1,
        attn_levels=[1],
        timesteps=20,
        sampling_timesteps=3,
    )
