"""Evaluate every trained model (best.pth) on its config's test set; write
results/comparison.{json,md}. Usage: python scripts/eval_all.py"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import build_dataset  # noqa: E402
from src.engine import evaluate_model  # noqa: E402
from src.metrics import build_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_config  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RUNS = {
    "SRCNN": "configs/srcnn_df2k_x4.yaml",
    "SRGAN": "configs/srgan_df2k_x4.yaml",
    "SwinIR": "configs/swinir_df2k_x4.yaml",
    "SR3": "configs/sr3_df2k_x4.yaml",
}


def eval_one(cfg_path):
    cfg = load_config(cfg_path)
    out = Path(f"experiments/{cfg.name}")
    ckpt = out / "best.pth" if (out / "best.pth").exists() else out / "last.pth"
    print(f"checkpoint: {ckpt}", flush=True)
    model = build_model(cfg.model).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict):
        for key in ("model", "params_ema", "params"):
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    datasets = {d.get("label", d.name): build_dataset(d) for d in cfg.test_datasets}
    metrics = build_metrics(cfg.get("metrics", ["psnr", "ssim", "lpips"]))
    return evaluate_model(model, datasets, metrics, DEVICE)


def main():
    results = {}
    for name, cfg_path in RUNS.items():
        print(f"\n===== EVAL {name} =====", flush=True)
        results[name] = eval_one(cfg_path)

    Path("results").mkdir(exist_ok=True)
    json.dump(results, open("results/comparison.json", "w"), indent=1)

    metrics = sorted({m for r in results.values() for s in r.values() for m in s})
    lines = ["| Model | Dataset | " + " | ".join(metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for name, r in results.items():
        for ds, scores in r.items():
            lines.append(f"| {name} | {ds} | " +
                         " | ".join(f"{scores.get(m, float('nan')):.4f}" for m in metrics) + " |")
    Path("results/comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nwrote results/comparison.{json,md}")


if __name__ == "__main__":
    main()
