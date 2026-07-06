"""No-reference (blind) image-quality metrics: NIQE, MUSIQ, CLIPIQA.

These score the SR image alone, without the HR ground truth -- useful because
PSNR/SSIM correlate poorly with perceived quality for generative SR. Backed by
``pyiqa`` (https://github.com/chaofengc/IQA-PyTorch); the underlying model is
created lazily on first call (MUSIQ/CLIPIQA download weights then), so importing
this module is cheap and needs no network.

Direction (for reading the results table):
  NIQE  lower is better
  MUSIQ, CLIPIQA  higher is better

Signature matches every metric: ``__call__(sr, hr=None)`` -- ``hr`` is accepted
(the Evaluator always passes it) but ignored.
"""

import torch

from . import METRICS
from .common import to_batch


class _PyIQAMetric:
    pyiqa_name = ""

    def __init__(self):
        self._model = None
        self._device = None

    def _ensure(self, ref: torch.Tensor):
        if self._model is None:
            import pyiqa  # lazy; see requirements.txt

            # NIQE uses float64 internally, which MPS does not support.
            # Run no-reference metrics on CPU when the evaluation tensor lives on MPS.
            self._device = (
                torch.device("cpu")
                if ref.device.type == "mps" and self.pyiqa_name == "niqe"
                else ref.device
            )
            self._model = pyiqa.create_metric(self.pyiqa_name, device=self._device)

    @torch.no_grad()
    def __call__(self, sr: torch.Tensor, hr: torch.Tensor = None) -> float:
        sr = to_batch(sr).clamp(0, 1)
        self._ensure(sr)
        return self._model(sr.to(self._device)).mean().item()


@METRICS.register("niqe")
class NIQE(_PyIQAMetric):
    pyiqa_name = "niqe"


@METRICS.register("musiq")
class MUSIQ(_PyIQAMetric):
    pyiqa_name = "musiq"


@METRICS.register("clipiqa")
class CLIPIQA(_PyIQAMetric):
    pyiqa_name = "clipiqa"
