"""Dataset registry.

Add a dataset by decorating a ``torch.utils.data.Dataset`` subclass with
``@DATASETS.register("YourName")``. All SR datasets return a dict
``{"lr": Tensor[C,h,w], "hr": Tensor[C,H,W], "name": str}`` with values in [0, 1].
"""

from ..utils.registry import Registry

DATASETS = Registry("datasets")

from . import div2k  # noqa: E402,F401
from . import benchmark  # noqa: E402,F401
from . import realsr  # noqa: E402,F401


def build_dataset(cfg):
    args = dict(cfg.get("args", {}) or {})
    return DATASETS.build(cfg.name, **args)


__all__ = ["DATASETS", "build_dataset"]
