import pytest
import torch
import torch.nn.functional as F

from src.datasets import transforms as T
from src.datasets.transforms import _jpeg


def test_bicubic_shape():
    assert T.make_lr(torch.rand(3, 32, 32), 4, degradation="bicubic").shape == (3, 8, 8)


def test_bicubic_downscale_is_antialiased():
    # The clean bicubic degradation must follow the standard SR protocol
    # (MATLAB/PIL-style antialiased downscale); a plain aliasing kernel is
    # out-of-distribution for models trained on standard bicubic LR.
    hr = torch.rand(3, 32, 32)
    expected = F.interpolate(hr.unsqueeze(0), size=(8, 8), mode="bicubic",
                             align_corners=False, antialias=True).squeeze(0).clamp(0, 1)
    assert torch.allclose(T.make_lr(hr, 4, degradation="bicubic"), expected)


def test_realistic_shape_and_range():
    lr = T.make_lr(torch.rand(3, 64, 64), 4, degradation="realistic")
    assert lr.shape == (3, 16, 16)
    assert lr.min() >= 0 and lr.max() <= 1


def test_realistic_pre_upscale_to_hr_size():
    lr = T.make_lr(torch.rand(3, 64, 64), 4, degradation="realistic", pre_upscale=True)
    assert lr.shape == (3, 64, 64)


def test_unknown_degradation_raises():
    with pytest.raises(ValueError):
        T.make_lr(torch.rand(3, 32, 32), 4, degradation="nope")


def test_jpeg_roundtrip_in_range():
    out = _jpeg(torch.rand(3, 16, 16), quality=50)
    assert out.shape == (3, 16, 16)
    assert out.min() >= 0 and out.max() <= 1
