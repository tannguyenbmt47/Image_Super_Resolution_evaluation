"""YAML config loading with attribute-style access.

Every experiment is described by a single YAML file (see ``configs/``).
``load_config`` returns a nested ``Config`` so code can write ``cfg.model.name``
instead of ``cfg["model"]["name"]``.

Limitation: ``Config`` subclasses ``dict``, so a key whose name collides with a
``dict`` method (``items``, ``keys``, ``values``, ``get``, ``copy``) is shadowed
on attribute access -- use indexing (``cfg["items"]``) for those. No config key in
this project uses such a name.
"""

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also exposes its keys as attributes (recursively)."""

    def __init__(self, data: dict):
        super().__init__()
        for key, value in data.items():
            self[key] = self._wrap(value)

    @staticmethod
    def _wrap(value: Any):
        if isinstance(value, dict):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(v) for v in value]
        return value

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any):
        self[name] = self._wrap(value)

    def get(self, key: str, default: Any = None):
        return self[key] if key in self else default


def load_config(path: str | Path) -> Config:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return Config(data or {})
