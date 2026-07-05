"""Evaluate a checkpoint and produce a shareable report bundle.

Like ``evaluate.py``, but besides the averaged table it saves every model input
and output and the per-image scores. Under ``--out`` (default
``results/<config name>``) it writes:

    images.zip             per test dataset: LR/ (model inputs) and SR/ (model
                           outputs) as lossless PNG
    report.md              setup + overall table + per-image metric tables
    per_image_metrics.csv  machine-readable per-image scores

    python scripts/evaluate_report.py --config configs/swinir_x4.yaml \
        --checkpoint weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
"""

import argparse
import csv
import io
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import build_dataset  # noqa: E402
from src.engine.evaluator import Evaluator  # noqa: E402
from src.metrics import build_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_config  # noqa: E402

# Reading the results table: which way is better, per metric.
DIRECTIONS = {"psnr": "higher", "ssim": "higher", "lpips": "lower",
              "niqe": "lower", "musiq": "higher", "clipiqa": "higher"}


def _png_bytes(t: torch.Tensor) -> bytes:
    from PIL import Image

    arr = t.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _md_table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="output dir (default results/<config name>)")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N images per dataset (quick checks)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out or f"results/{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(cfg.model).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    # Unwrap: this repo's Trainer saves {"model": ...}; official release
    # checkpoints (e.g. SwinIR) use {"params": ...} or {"params_ema": ...}.
    for key in ("model", "params_ema", "params"):
        if key in state:
            state = state[key]
            break
    model.load_state_dict(state)

    datasets = {d.get("label", d.name): build_dataset(d) for d in cfg.test_datasets}
    if args.limit:
        from torch.utils.data import Subset
        datasets = {k: Subset(v, range(min(args.limit, len(v)))) for k, v in datasets.items()}
    metrics = build_metrics(cfg.get("metrics", ["psnr", "ssim", "lpips"]))
    metric_names = list(metrics)
    evaluator = Evaluator(metrics, args.device)

    rows = []  # (dataset, image, {metric: value})
    overall = {}
    # PNGs are already compressed; ZIP_STORED avoids pointless re-deflating.
    with zipfile.ZipFile(out / "images.zip", "w", zipfile.ZIP_STORED) as zf:
        for ds_name, ds in datasets.items():
            def on_sample(name, lr, sr, scores, _ds=ds_name):
                zf.writestr(f"{_ds}/LR/{name}.png", _png_bytes(lr))
                zf.writestr(f"{_ds}/SR/{name}.png", _png_bytes(sr))
                rows.append((_ds, name, scores))
            overall[ds_name] = evaluator.run(model, ds, ds_name, on_sample=on_sample)

    with open(out / "per_image_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "image", *metric_names])
        for ds_name, name, scores in rows:
            writer.writerow([ds_name, name, *[f"{scores[m]:.6f}" for m in metric_names]])

    n_params = sum(p.numel() for p in model.parameters())
    device_name = (torch.cuda.get_device_name(0)
                   if args.device.startswith("cuda") and torch.cuda.is_available()
                   else args.device)
    header_cells = [f"{m} ({DIRECTIONS.get(m, '?')} is better)" for m in metric_names]
    report = [
        f"# Evaluation report: {cfg.name}",
        "",
        f"- Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: `{cfg.model.name}` ({n_params / 1e6:.2f}M parameters)",
        f"- Checkpoint: `{Path(args.checkpoint).name}`",
        f"- Config: `{Path(args.config).name}`",
        f"- Device: {device_name} (torch {torch.__version__})",
        "- Protocol: LR inputs are synthesized from the HR originals per the dataset"
        " config (antialiased bicubic downscale unless the config says otherwise);"
        " SR outputs are the raw model outputs clamped to [0, 1].",
        "- Images: `images.zip` holds, per dataset, the model inputs (`LR/`) and"
        " outputs (`SR/`) as lossless PNG. HR references are the original dataset"
        " images and are not duplicated here.",
        "",
        "## Overall results",
        "",
        _md_table(["dataset", "images", *header_cells],
                  [[ds, sum(1 for r in rows if r[0] == ds),
                    *[f"{overall[ds][m]:.4f}" for m in metric_names]] for ds in overall]),
    ]
    for ds_name in overall:
        report += [
            "",
            f"## Per-image results — {ds_name}",
            "",
            _md_table(["image", *metric_names],
                      [[name, *[f"{scores[m]:.4f}" for m in metric_names]]
                       for ds, name, scores in rows if ds == ds_name]),
        ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    zip_mb = (out / "images.zip").stat().st_size / 2**20
    print(f"wrote {out / 'report.md'}")
    print(f"wrote {out / 'per_image_metrics.csv'} ({len(rows)} rows)")
    print(f"wrote {out / 'images.zip'} ({zip_mb:.0f} MiB)")


if __name__ == "__main__":
    main()
