"""SwinIR (Liang et al., 2021) -- transformer SR with shifted-window attention.

Thin registry adapter over the vendored official architecture
(``swinir_arch.py``), mapping this project's ``scale``/``num_channels``
conventions onto the official constructor. Defaults are the classical-SR
SwinIR-M preset, so the authors' released DF2K x4 checkpoint
(``001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth``) loads key-for-key -- unlike
SRCNN/SR3, SwinIR is meant to be *evaluated with official weights* rather than
trained here (though as a plain feed-forward ``forward(lr) -> sr`` model the
Trainer can also fine-tune it with a pixel loss).

Usage notes:
- SwinIR upscales internally, so its datasets must set ``pre_upscale: false``.
- Arbitrary eval image sizes work: ``forward`` pads the input to a multiple of
  ``window_size`` (reflect) and crops the output back.
- The s48 DIV2K official checkpoints need ``img_size: 48`` in ``model.args``
  (the attention-mask buffers in the checkpoint depend on the training patch
  size); the default ``img_size: 64`` matches the s64 DF2K releases.
"""

from . import MODELS
from .swinir_arch import SwinIR as _SwinIRArch

# Classical-SR "M" preset -- matches the official 001_classicalSR_DF2K_s64w8
# release checkpoints (any of x2/x3/x4 via ``scale``).
_CLASSICAL_M = dict(
    img_size=64,
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler="pixelshuffle",
    resi_connection="1conv",
)


@MODELS.register("SwinIR")
class SwinIR(_SwinIRArch):
    def __init__(self, scale: int = 4, num_channels: int = 3, **overrides):
        args = {**_CLASSICAL_M, **overrides}
        super().__init__(upscale=scale, in_chans=num_channels, **args)
        self.scale = scale
