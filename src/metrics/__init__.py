"""Metric registry.

A metric is a callable ``(sr, hr) -> float`` where ``sr`` and ``hr`` are
tensors [C, H, W] or [N, C, H, W] in [0, 1]. Register with
``@METRICS.register("YourMetric")``. ``build_metrics`` turns a config list like
``["psnr", "ssim"]`` (or ``[{name: psnr, args: {...}}]``) into ready callables.
"""

from ..utils.registry import Registry

METRICS = Registry("metrics")

from . import psnr  # noqa: E402,F401
from . import ssim  # noqa: E402,F401
from . import lpips_metric  # noqa: E402,F401
from . import noref  # noqa: E402,F401  NIQE / MUSIQ / CLIPIQA (no-reference)


def build_metrics(cfg_list):
    """Return ``{name: callable}`` for a list of metric configs."""
    metrics = {}
    for item in cfg_list:
        if isinstance(item, str):
            name, args = item, {}
        else:
            name, args = item["name"], dict(item.get("args", {}) or {})
        metrics[name] = METRICS.build(name, **args)
    return metrics


__all__ = ["METRICS", "build_metrics"]
