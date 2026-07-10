import torch

from src.models import MODELS, build_model
from src.utils import Config


def test_expected_models_registered():
    assert set(MODELS.names()) == {"SRCNN", "SR3", "SRGAN", "SwinIR"}


def test_srcnn_preserves_size():
    # SRCNN consumes a pre-upscaled LR, so output size == input size
    m = build_model(Config({"name": "SRCNN", "args": {"scale": 4}}))
    x = torch.rand(2, 3, 32, 32)
    assert m(x).shape == x.shape


def _tiny_swinir():
    # shrunken SwinIR so the forward pass is fast on CPU
    return build_model(Config({"name": "SwinIR", "args": {
        "scale": 4, "img_size": 16, "window_size": 4,
        "embed_dim": 16, "depths": [2, 2], "num_heads": [2, 2],
    }}))


def test_swinir_upscales():
    m = _tiny_swinir()
    x = torch.rand(1, 3, 16, 16)
    assert m(x).shape == (1, 3, 64, 64)


def test_swinir_handles_non_window_multiple():
    # eval images have arbitrary sizes; SwinIR pads to a window multiple
    # internally and crops the output back
    m = _tiny_swinir()
    x = torch.rand(1, 3, 10, 14)
    assert m(x).shape == (1, 3, 40, 56)
