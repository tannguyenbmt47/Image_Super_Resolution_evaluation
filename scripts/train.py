"""Train an SR model from a YAML config.

python scripts/train.py --config configs/srcnn_df2k_x4.yaml
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import build_dataset  # noqa: E402
from src.engine import Trainer  # noqa: E402
from src.metrics import build_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_config  # noqa: E402


def _label(d):
    """Display name for a dataset config (``label`` if set, else registry name)."""
    return d.get("label", d.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )
    parser.add_argument("--out", default=None, help="override output dir")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="checkpoint to initialise from (stage-2 fine-tuning)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="override train.epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="override train.batch_size"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="override train.num_workers"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="override train_dataset.args.repeat (use 1 for quick runs)",
    )
    parser.add_argument(
        "--no-val", action="store_true", help="skip in-training validation"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.out or f"experiments/{cfg.get('name', Path(args.config).stem)}"
    if args.epochs is not None:
        cfg.train["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.train["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg.train["num_workers"] = args.num_workers
    if args.repeat is not None:
        cfg.train_dataset.args["repeat"] = args.repeat

    model = build_model(cfg.model)
    if args.pretrained:
        state = torch.load(args.pretrained, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        print(f"initialised from {args.pretrained}")

    train_set = build_dataset(cfg.train_dataset)
    val_sets = (
        {}
        if args.no_val
        else {_label(d): build_dataset(d) for d in cfg.get("val_datasets", [])}
    )
    # keep validation light; heavy no-reference metrics belong in evaluate.py
    metrics = build_metrics(cfg.get("val_metrics", ["psnr", "ssim"]))

    trainer = Trainer(
        model,
        train_set,
        cfg.train,
        device=args.device,
        val_sets=val_sets,
        metrics=metrics,
        out_dir=out_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
