"""No-reference metrics: registration + lazy construction (no network needed).
Actually computing them requires pyiqa + weights -- see the integration test."""

from src.metrics import METRICS, build_metrics


def test_noref_metrics_registered():
    for name in ("niqe", "musiq", "clipiqa"):
        assert name in METRICS


def test_noref_metrics_build_without_network():
    # constructing the wrappers must not import pyiqa or download anything
    metrics = build_metrics(["niqe", "musiq", "clipiqa"])
    assert set(metrics) == {"niqe", "musiq", "clipiqa"}
    assert all(m._model is None for m in metrics.values())
