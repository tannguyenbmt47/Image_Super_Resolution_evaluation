import torch

from src.metrics import build_metrics
from src.metrics.common import crop_border, rgb_to_y
from src.metrics.psnr import PSNR
from src.metrics.ssim import SSIM


def test_psnr_identical_is_high():
    x = torch.rand(3, 32, 32)
    assert PSNR(crop=2)(x, x) > 80


def test_ssim_identical_is_one():
    x = torch.rand(3, 32, 32)
    assert abs(SSIM(crop=2)(x, x) - 1.0) < 1e-3


def test_psnr_orders_by_quality():
    hr = torch.rand(3, 32, 32)
    near = (hr + 0.01 * torch.randn_like(hr)).clamp(0, 1)
    far = (hr + 0.30 * torch.randn_like(hr)).clamp(0, 1)
    psnr = PSNR(crop=2)
    assert psnr(near, hr) > psnr(far, hr)


def test_ssim_orders_by_quality():
    hr = torch.rand(3, 32, 32)
    near = (hr + 0.01 * torch.randn_like(hr)).clamp(0, 1)
    far = (hr + 0.30 * torch.randn_like(hr)).clamp(0, 1)
    ssim = SSIM(crop=2)
    assert ssim(near, hr) > ssim(far, hr)


def test_build_metrics_mixed_spec():
    m = build_metrics([{"name": "psnr", "args": {"crop": 2}}, "ssim"])
    assert set(m) == {"psnr", "ssim"}


def test_metric_accepts_batched_and_unbatched():
    psnr = PSNR(crop=2)
    a, b = torch.rand(3, 16, 16), torch.rand(3, 16, 16)
    assert isinstance(psnr(a, b), float)
    assert isinstance(psnr(a.unsqueeze(0), b.unsqueeze(0)), float)


def test_rgb_to_y_shape():
    assert rgb_to_y(torch.rand(2, 3, 8, 8)).shape == (2, 1, 8, 8)


def test_crop_border():
    x = torch.rand(1, 1, 10, 10)
    assert crop_border(x, 2).shape == (1, 1, 6, 6)
    assert crop_border(x, 0).shape == (1, 1, 10, 10)
