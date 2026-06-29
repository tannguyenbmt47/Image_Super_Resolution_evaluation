import torch

from src.datasets import transforms as T


def test_make_lr_downscales():
    hr = torch.rand(3, 32, 32)
    assert T.make_lr(hr, scale=4).shape == (3, 8, 8)


def test_make_lr_pre_upscale_keeps_hr_size():
    hr = torch.rand(3, 32, 32)
    assert T.make_lr(hr, scale=4, pre_upscale=True).shape == (3, 32, 32)


def test_make_lr_in_range():
    hr = torch.rand(3, 32, 32)
    lr = T.make_lr(hr, scale=4)
    assert lr.min() >= 0 and lr.max() <= 1


def test_paired_random_crop_size_aligned():
    hr = torch.rand(3, 50, 40)
    crop = T.paired_random_crop(hr, scale=4, patch=22)
    # patch is floored to a multiple of scale: 22 -> 20
    assert crop.shape == (3, 20, 20)


def test_augment_preserves_square_shape():
    hr = torch.rand(3, 16, 16)
    assert T.augment(hr).shape == (3, 16, 16)
