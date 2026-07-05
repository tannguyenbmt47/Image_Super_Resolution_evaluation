"""Evaluate a trained checkpoint across test datasets and metrics.

    python scripts/evaluate.py --config configs/sr3_df2k_x4.yaml \
        --checkpoint experiments/sr3_df2k_x4/last.pth
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.nn import DataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import build_dataset  # noqa: E402
from src.engine import evaluate_model  # noqa: E402
from src.metrics import build_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_config  # noqa: E402


def print_table(results: dict):
    metric_names = sorted({m for s in results.values() for m in s})
    header = f"{'dataset':<14}" + "".join(f"{m:>10}" for m in metric_names)
    print(header)
    print("-" * len(header))
    for ds, scores in results.items():
        row = f"{ds:<14}" + "".join(f"{scores.get(m, float('nan')):>10.4f}" for m in metric_names)
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--device",
        default=(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N images per dataset (quick checks)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg.model).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)
    if args.device.startswith("cuda") and torch.cuda.device_count() > 1:
        model = DataParallel(model)

    datasets = {d.get("label", d.name): build_dataset(d) for d in cfg.test_datasets}
    if args.limit:
        from torch.utils.data import Subset
        datasets = {k: Subset(v, range(min(args.limit, len(v)))) for k, v in datasets.items()}
    metrics = build_metrics(cfg.get("metrics", ["psnr", "ssim", "lpips"]))

    results = evaluate_model(model, datasets, metrics, args.device)
    print_table(results)


if __name__ == "__main__":
    main()
