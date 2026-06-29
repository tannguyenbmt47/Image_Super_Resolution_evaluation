"""End-to-end: build datasets -> train one epoch -> evaluate, for both a
feed-forward model (pixel-loss path) and a diffusion model (compute_loss /
super_resolve path). Exercises the engine's model-type dispatch."""

import pytest

from src.datasets import build_dataset
from src.engine import Trainer, evaluate_model
from src.metrics import build_metrics
from src.models import build_model
from src.utils import Config


def _make_sets(hr_dir, pre_upscale=False):
    train = build_dataset(Config({"name": "DIV2K", "args": {
        "hr_dir": hr_dir, "scale": 4, "patch_size": 16, "repeat": 2,
        "pre_upscale": pre_upscale}}))
    test = build_dataset(Config({"name": "Benchmark", "args": {
        "hr_dir": hr_dir, "scale": 4, "pre_upscale": pre_upscale, "name": "t"}}))
    return train, test


def _train_cfg():
    return Config({"epochs": 1, "batch_size": 2, "num_workers": 0,
                   "lr": 1e-3, "val_every": 1})


@pytest.mark.integration
def test_feedforward_train_and_eval(hr_dir, tmp_path):
    # SRCNN consumes a pre-upscaled LR, so datasets use pre_upscale=True
    train, test = _make_sets(hr_dir, pre_upscale=True)
    model = build_model(Config({"name": "SRCNN", "args": {"scale": 4}}))
    metrics = build_metrics(["psnr", "ssim"])
    out = tmp_path / "run"

    Trainer(model, train, _train_cfg(), device="cpu",
            val_sets={"t": test}, metrics=metrics, out_dir=str(out)).train()

    assert (out / "last.pth").exists()
    assert (out / "best.pth").exists()  # saved when validation metric improves
    results = evaluate_model(model, {"t": test}, metrics, device="cpu")
    assert set(results["t"]) == {"psnr", "ssim"}
    assert all(isinstance(v, float) for v in results["t"].values())


@pytest.mark.integration
def test_diffusion_train_and_eval(hr_dir, tmp_path, tiny_diffusion_args):
    train, test = _make_sets(hr_dir)
    model = build_model(Config({"name": "SR3", "args": {
        "scale": 4, **tiny_diffusion_args}}))
    metrics = build_metrics(["psnr"])
    out = tmp_path / "run"

    Trainer(model, train, _train_cfg(), device="cpu",
            val_sets={"t": test}, metrics=metrics, out_dir=str(out)).train()

    assert (out / "last.pth").exists()
    results = evaluate_model(model, {"t": test}, metrics, device="cpu")
    assert "psnr" in results["t"]


@pytest.mark.integration
def test_early_stopping(hr_dir, tmp_path):
    """Drive the early-stop decision directly with crafted validation scores."""
    train, _ = _make_sets(hr_dir, pre_upscale=True)
    model = build_model(Config({"name": "SRCNN", "args": {"scale": 4}}))
    cfg = Config({"epochs": 5, "batch_size": 2, "num_workers": 0, "lr": 1e-3,
                  "val_every": 1, "early_stop_patience": 2, "early_stop_metric": "psnr"})
    out = tmp_path / "run"
    t = Trainer(model, train, cfg, device="cpu", out_dir=str(out))

    t._update_early_stop({"t": {"psnr": 20.0}})   # first -> best, saves best.pth
    assert (out / "best.pth").exists()
    assert not t.should_stop

    t._update_early_stop({"t": {"psnr": 19.0}})   # worse -> no_improve = 1
    assert not t.should_stop

    t._update_early_stop({"t": {"psnr": 19.5}})   # still below best -> no_improve = 2
    assert t.should_stop                          # patience reached


@pytest.mark.integration
def test_early_stop_metric_direction(hr_dir, tmp_path):
    train, _ = _make_sets(hr_dir, pre_upscale=True)
    model = build_model(Config({"name": "SRCNN", "args": {"scale": 4}}))
    # niqe is lower-is-better, so improvement means a smaller value
    cfg = Config({"epochs": 1, "batch_size": 2, "num_workers": 0, "lr": 1e-3,
                  "early_stop_metric": "niqe"})
    t = Trainer(model, train, cfg, device="cpu", out_dir=str(tmp_path / "run"))
    assert t.monitor_mode == "min"
    t._update_early_stop({"t": {"niqe": 5.0}})
    t._update_early_stop({"t": {"niqe": 4.0}})    # lower = better -> improvement
    assert t.no_improve == 0
