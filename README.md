# HMUD — Single Image Super-Resolution Research

A small, extensible PyTorch framework for comparing **multiple SR models** across
**multiple datasets** under **multiple metrics**. Everything is config-driven and
registry-based: adding a new model, dataset, or metric is a one-file change.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> All commands use the project virtual environment at `.venv/`.

## Datasets

Datasets are not committed (see `.gitignore`). Download the core ones with:

```bash
bash scripts/download_data.sh        # DIV2K train+val + benchmark sets (~4GB)
```

Layout under `data/`:

```
data/
  DIV2K/DIV2K_train_HR/        # pretrain (synthetic LR-HR)
  DIV2K/DIV2K_valid_HR/        # DIV2K-Val evaluation
  Flickr2K/Flickr2K_HR/        # optional, for DF2K (~24GB, not auto-downloaded)
  benchmark/{Set5,Set14,BSD100,Urban100,Manga109}/HR/
  RealSR/{Train,Test}/{LR,HR}/ # real-world pairs, manual download (stage-2)
```

For the synthetic sets only **HR** images are needed — LR is generated on the fly,
either by **bicubic** downscaling (classical SR) or a **realistic** Real-ESRGAN-style
degradation (`degradation: realistic` in the config). RealSR uses *real* LR/HR pairs.

**Training plan:** stage 1 pretrains on DF2K (DIV2K train + Flickr2K) synthetic pairs;
stage 2 fine-tunes on the RealSR train split. Evaluate on DIV2K-Val + RealSR.

```bash
# stage 1
.venv/bin/python scripts/train.py --config configs/srcnn_df2k_x4.yaml
.venv/bin/python scripts/train.py --config configs/sr3_df2k_x4.yaml
# stage 2 (fine-tune from stage 1): point train_dataset at RealSR and pass
# --pretrained <stage-1 checkpoint>
```

## Train

```bash
.venv/bin/python scripts/train.py --config configs/srcnn_df2k_x4.yaml
```

Checkpoints and validation logs go to `experiments/<name>/`.

## Evaluate

```bash
.venv/bin/python scripts/evaluate.py \
    --config configs/srcnn_df2k_x4.yaml \
    --checkpoint experiments/srcnn_df2k_x4/last.pth
```

Prints a `dataset × metric` table.

## Models

| Model   | Type                  | Notes                                              |
|---------|-----------------------|----------------------------------------------------|
| SRCNN   | feed-forward (pixel)  | classic 3-layer baseline; needs `pre_upscale: true`|
| SR3     | conditional diffusion | diffuses HR conditioned on up(LR)                  |

SR3 trains on a noise-prediction loss and produces SR by iterative DDIM sampling —
evaluation is much slower than the feed-forward SRCNN. Tune `sampling_timesteps`
in the config to trade speed for quality.

## Metrics

| Metric  | Type            | Direction |
|---------|-----------------|-----------|
| PSNR    | full-reference  | ↑         |
| SSIM    | full-reference  | ↑         |
| LPIPS   | full-reference  | ↓         |
| NIQE    | no-reference    | ↓         |
| MUSIQ   | no-reference    | ↑         |
| CLIPIQA | no-reference    | ↑         |

PSNR/SSIM are computed on the Y channel with a `scale`-pixel border crop; LPIPS on
RGB. No-reference metrics (via `pyiqa`) score the SR image without ground truth.
