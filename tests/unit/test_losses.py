import pytest
import torch

from src.losses import CharbonnierLoss, build_loss
from src.utils import Config


def test_charbonnier_small_for_identical():
    x = torch.rand(2, 3, 16, 16)
    assert CharbonnierLoss()(x, x).item() < 1e-2


def test_charbonnier_positive_for_different():
    x, y = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    assert CharbonnierLoss()(x, y).item() > 0


def test_build_loss_defaults_to_l1():
    crit = build_loss(Config({}))
    x = torch.rand(1, 3, 8, 8)
    assert crit.perceptual is None
    assert crit(x, x).item() == 0.0


def test_build_loss_charbonnier():
    crit = build_loss(Config({"loss": "charbonnier"}))
    x = torch.rand(1, 3, 8, 8)
    assert crit(x, x).item() < 1e-2


def test_build_loss_unknown_pixel_raises():
    with pytest.raises(ValueError):
        build_loss(Config({"loss": "nope"}))


def test_perceptual_disabled_when_weight_zero():
    # weight 0 must not construct VGG (which would need a network download)
    crit = build_loss(Config({"loss": "l1", "perceptual_weight": 0.0}))
    assert crit.perceptual is None
