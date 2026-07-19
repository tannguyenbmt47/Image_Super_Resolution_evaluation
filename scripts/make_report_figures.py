"""Generate the three report figures into images/:

  training-loss.png     4 panels, one per model (loss scales differ)
  val-psnr-ssim.png     shared validation PSNR/SSIM curves (image-epoch axis)
  qualitative-grid.png  rows = val images; cols = LR/Bicubic/4 models/GT

Usage: .venv/bin/python scripts/make_report_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import build_dataset  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import Config, load_config  # noqa: E402

OUT = Path("images")
OUT.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "SRCNN": "configs/srcnn_df2k_x4.yaml",
    "SRGAN": "configs/srgan_df2k_x4.yaml",
    "SwinIR": "configs/swinir_df2k_x4.yaml",
    "SR3": "configs/sr3_df2k_x4.yaml",
}
# 1 tile-epoch (30581 tiles) ~ 8.86 image-epochs (3450 source images)
EPOCH_SCALE = 30581 / 3450


def history(cfg_path):
    cfg = load_config(cfg_path)
    return json.load(open(f"experiments/{cfg.name}/history.json"))


def fig_training_loss():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (name, cfg_path) in zip(axes, MODELS.items()):
        h = history(cfg_path)["train_loss"]
        ep = [e["epoch"] * EPOCH_SCALE for e in h]
        ax.plot(ep, [e["loss"] for e in h], marker="o", color="tab:blue")
        ax.set_title(f"{name}")
        ax.set_xlabel("image-epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("training loss")
    plt.tight_layout()
    plt.savefig(OUT / "training-loss.png", dpi=200)
    plt.close()
    print("wrote images/training-loss.png")


def fig_val_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, cfg_path in MODELS.items():
        h = history(cfg_path)["val"]
        for ax, metric in zip(axes, ["psnr", "ssim"]):
            pts = [(v["epoch"] * EPOCH_SCALE, s[metric]) for v in h
                   for ds, s in v.items() if isinstance(s, dict) and metric in s]
            if pts:
                ax.plot(*zip(*pts), marker="s", label=name)
    axes[0].set_title("Validation PSNR (DIV2K-Val crop 256)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation SSIM (DIV2K-Val crop 256)")
    axes[1].set_ylabel("SSIM")
    for ax in axes:
        ax.set_xlabel("image-epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "val-psnr-ssim.png", dpi=200)
    plt.close()
    print("wrote images/val-psnr-ssim.png")


@torch.no_grad()
def fig_qualitative(indices=(3, 17, 46)):
    ds = build_dataset(Config({"name": "Benchmark", "args": {
        "hr_dir": "data/DIV2K/DIV2K_valid_HR", "scale": 4, "max_hr": 256}}))

    # load the four trained models (best.pth)
    nets = {}
    for name, cfg_path in MODELS.items():
        cfg = load_config(cfg_path)
        m = build_model(cfg.model).to(DEVICE).eval()
        state = torch.load(f"experiments/{cfg.name}/best.pth",
                           map_location=DEVICE, weights_only=False)
        m.load_state_dict(state["model"] if "model" in state else state, strict=False)
        nets[name] = m

    def to_img(t):
        return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    cols = ["LR", "Bicubic", "SRCNN", "SRGAN", "SwinIR", "SR3", "Ground truth"]
    fig, axes = plt.subplots(len(indices), len(cols),
                             figsize=(2.1 * len(cols), 2.15 * len(indices)))
    for r, idx in enumerate(indices):
        s = ds[idx]
        hr = s["hr"].to(DEVICE)
        lr = s["lr"].to(DEVICE)
        size = hr.shape[-2:]
        up = F.interpolate(lr.unsqueeze(0), size=size, mode="bicubic",
                           align_corners=False)[0].clamp(0, 1)
        outs = {"LR": F.interpolate(lr.unsqueeze(0), size=size, mode="nearest")[0],
                "Bicubic": up, "Ground truth": hr}
        for name, m in nets.items():
            # SRCNN consumes the pre-upscaled LR; others take raw LR
            inp = up if name == "SRCNN" else lr
            infer = m.super_resolve if hasattr(m, "super_resolve") else m
            outs[name] = infer(inp.unsqueeze(0))[0].clamp(0, 1)
        for c, col in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(to_img(outs[col]))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(col, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "qualitative-grid.png", dpi=200)
    plt.close()
    print("wrote images/qualitative-grid.png")


if __name__ == "__main__":
    fig_training_loss()
    fig_val_curves()
    fig_qualitative()
