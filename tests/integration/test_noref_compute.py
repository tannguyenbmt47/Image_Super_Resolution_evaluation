"""Actually compute a no-reference metric. Requires pyiqa and (first run) a
weights download, so it is an integration test and skips if unavailable."""

import pytest
import torch


@pytest.mark.integration
def test_niqe_produces_a_score():
    pytest.importorskip("pyiqa")
    from src.metrics import build_metrics

    niqe = build_metrics(["niqe"])["niqe"]
    try:
        score = niqe(torch.rand(3, 96, 96))
    except Exception as exc:  # offline / weights missing
        pytest.skip(f"pyiqa niqe unavailable: {exc}")
    assert isinstance(score, float)
