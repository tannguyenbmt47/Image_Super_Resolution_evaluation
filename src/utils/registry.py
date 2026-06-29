"""A tiny name -> factory registry.

This is the backbone of the project: models, datasets and metrics each own a
Registry instance. New components are added by decorating a class/function with
``@SOME_REGISTRY.register()`` -- no central if/else dispatch to edit.
"""

from typing import Callable, Dict, Optional


class Registry:
    def __init__(self, name: str):
        self._name = name
        self._items: Dict[str, Callable] = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """Decorator. Registers the wrapped object under ``name`` (or its
        ``__name__`` if omitted)."""

        def decorator(obj: Callable) -> Callable:
            key = name or getattr(obj, "__name__")
            if key in self._items:
                raise KeyError(f"'{key}' already registered in {self._name} registry")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(
                f"'{name}' is not registered in {self._name} registry. "
                f"Available: {sorted(self._items)}"
            )
        return self._items[name]

    def build(self, name: str, /, **kwargs):
        """Instantiate the registered object identified by ``name``.

        ``name`` is positional-only so components may themselves take a ``name``
        keyword argument (e.g. the Benchmark dataset) without colliding."""
        return self.get(name)(**kwargs)

    def names(self):
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items

    def __repr__(self) -> str:
        return f"Registry({self._name}, items={self.names()})"
