"""Model registry.

Add a new architecture by creating a module here and decorating its factory
with ``@MODELS.register("YourName")``. Importing this package triggers
registration of every model module below.
"""

from ..utils.registry import Registry

MODELS = Registry("models")

# Import submodules so their @MODELS.register() decorators run.
from . import srcnn  # noqa: E402,F401  feed-forward
from . import sr3  # noqa: E402,F401    diffusion
from . import srgan  # noqa: E402,F401  GAN
from . import swinir  # noqa: E402,F401 feed-forward transformer (official arch)


def build_model(cfg):
    """Build a model from a config node like ``{name: SR3, args: {...}}``."""
    args = dict(cfg.get("args", {}) or {})
    return MODELS.build(cfg.name, **args)


__all__ = ["MODELS", "build_model"]
