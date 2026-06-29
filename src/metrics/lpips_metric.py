"""LPIPS perceptual distance (lower is better).

Wraps the ``lpips`` package (Zhang et al., CVPR 2018). The network is loaded
lazily on first call so the package is only required when this metric is used.
LPIPS expects RGB in [-1, 1]; we convert from our [0, 1] convention.
"""

import torch

from . import METRICS
from .common import to_batch


@METRICS.register("lpips")
class LPIPS:
    def __init__(self, net: str = "alex", device: str | None = None):
        self.net = net
        self.device = device
        self._model = None

    def _ensure_model(self, ref: torch.Tensor):
        if self._model is None:
            import lpips  # imported lazily; see requirements.txt

            self._model = lpips.LPIPS(net=self.net)
            self._model.to(self.device or ref.device).eval()

    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor) -> float:
        sr, hr = to_batch(sr).clamp(0, 1), to_batch(hr).clamp(0, 1)
        self._ensure_model(sr)
        dev = next(self._model.parameters()).device
        sr, hr = sr.to(dev) * 2 - 1, hr.to(dev) * 2 - 1
        return self._model(sr, hr).mean().item()
