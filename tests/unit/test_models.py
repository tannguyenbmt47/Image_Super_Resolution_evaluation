import torch

from src.models import MODELS, build_model
from src.utils import Config


def test_expected_models_registered():
    assert set(MODELS.names()) == {"SRCNN", "SR3"}


def test_srcnn_preserves_size():
    # SRCNN consumes a pre-upscaled LR, so output size == input size
    m = build_model(Config({"name": "SRCNN", "args": {"scale": 4}}))
    x = torch.rand(2, 3, 32, 32)
    assert m(x).shape == x.shape
