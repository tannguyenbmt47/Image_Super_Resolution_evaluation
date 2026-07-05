# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A PyTorch research framework for **Single Image Super-Resolution**: train and
compare several SR models across several benchmark datasets under several quality
metrics. The design goal is that adding a model/dataset/metric never requires
touching dispatch code — only adding one file with one decorator.

## Environment & commands

Always use the project virtual environment `.venv/` — never system Python.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# download datasets into ./data (DIV2K train+val, Flickr2K for DF2K)
bash scripts/download_data.sh

# train (config drives everything: model, datasets, metrics, schedule)
.venv/bin/python scripts/train.py --config configs/srcnn_df2k_x4.yaml [--device cpu] [--out DIR]

# stage-2 fine-tuning: initialise from a pretrained checkpoint
.venv/bin/python scripts/train.py --config <realsr-finetune-config> \
    --pretrained experiments/srcnn_df2k_x4/last.pth

# evaluate a checkpoint -> prints dataset × metric table
.venv/bin/python scripts/evaluate.py --config configs/sr3_df2k_x4.yaml --checkpoint experiments/sr3_df2k_x4/last.pth

# SwinIR is evaluation-only with the authors' released weights (no training here)
bash scripts/download_weights.sh
.venv/bin/python scripts/evaluate.py --config configs/swinir_x4.yaml \
    --checkpoint weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth

# like evaluate.py, but also writes results/<name>/: report.md (per-image +
# overall metrics), per_image_metrics.csv, images.zip (LR inputs / SR outputs)
.venv/bin/python scripts/evaluate_report.py --config configs/swinir_x4.yaml \
    --checkpoint weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth
```

Tests (`pytest`, all CPU, ~5s):

```bash
.venv/bin/python -m pytest                      # everything
.venv/bin/python -m pytest -m "not integration" # unit only
.venv/bin/python -m pytest -m integration       # end-to-end only
```

`tests/unit/` covers registry, config, metrics, transforms, model shapes, and the
diffusion core (UNet/q_sample/sampling). `tests/integration/` builds datasets from
synthetic PNGs (the `hr_dir` fixture in `tests/conftest.py`) and runs a full
train→eval cycle for both a feed-forward and a diffusion model. Diffusion tests use
the tiny-net `tiny_diffusion_args` fixture to stay fast. `pytest.ini` sets
`pythonpath = .` so `import src` works without installation.

## Architecture — the registry pattern

Three registries are the backbone (`src/utils/registry.py`). Each of models,
datasets, and metrics owns a `Registry` instance. Components self-register via a
decorator; the package `__init__` imports every submodule so those decorators run;
a `build_*` helper turns a config node into an instance. **To add a component:
create a file, decorate it, done — no central if/else to edit.**

- `src/models/__init__.py` → `MODELS`, `build_model(cfg.model)`
- `src/datasets/__init__.py` → `DATASETS`, `build_dataset(cfg)`
- `src/metrics/__init__.py` → `METRICS`, `build_metrics(cfg_list)`

`Registry.build(name, /, **kwargs)` takes `name` **positional-only** on purpose, so
a component may itself have a `name` constructor kwarg (the Benchmark dataset does)
without colliding. Keep it that way.

## Cross-cutting conventions (read before editing models/datasets/metrics)

These contracts are what let any model run on any dataset under any metric:

- **Tensor convention:** images are float tensors in `[0, 1]`, channel-first
  `[C, H, W]` (or batched `[N, C, H, W]`), RGB.
- **Dataset output:** every dataset `__getitem__` returns
  `{"lr": Tensor, "hr": Tensor, "name": str}`. LR is produced by bicubic
  downscaling of HR (`src/datasets/transforms.py: make_lr`); datasets need only HR
  images on disk.
- **`pre_upscale` flag:** SRCNN consumes an LR pre-upscaled to HR size, so its
  configs set `pre_upscale: true` on **every** dataset. SR3 upscales internally and
  sets `pre_upscale: false`. A model and its datasets must agree on this flag or
  shapes won't line up.
- **Model interface (feed-forward vs diffusion):** the engine dispatches on two
  optional methods. Feed-forward models (SRCNN) define only `forward(lr) -> sr`
  and are trained with a pixel loss. Diffusion models (SR3) define
  `compute_loss(lr, hr) -> scalar` (noise-prediction loss) and
  `super_resolve(lr) -> sr` (iterative sampling); the Trainer/Evaluator call those
  when present. A new model only needs the pair that matches its paradigm.
- **Metric signature:** a metric is a callable `(sr, hr) -> float`. Full-reference
  metrics (PSNR/SSIM/LPIPS) compare against HR; PSNR/SSIM default to Y-channel with a
  `crop` border (set `crop` = scale factor, standard SR practice), LPIPS on RGB.
  No-reference metrics (NIQE/MUSIQ/CLIPIQA, `src/metrics/noref.py`) score the SR image
  alone and ignore `hr`. LPIPS and the no-reference metrics lazy-load their backends
  (`lpips`, `pyiqa`) on first call, so importing is cheap and offline.

## Datasets, degradation & two-stage training

- **Degradation** (`src/datasets/transforms.py: make_lr`): LR is synthesized from HR
  by either `degradation: bicubic` (classical, clean) or `degradation: realistic`
  (Real-ESRGAN-style blur→downsample→noise→JPEG). Set it per dataset in the config.
- **DF2K pretrain**: the `DIV2K` dataset's `hr_dir` accepts a **list** of folders, so
  DF2K = `[DIV2K_train_HR, Flickr2K_HR]`. Missing folders are skipped (Flickr2K is
  optional), so a DF2K config runs with only DIV2K present.
- **DIV2K-Val / benchmarks**: use the `Benchmark` dataset (folder of HR; LR synthesized)
  pointed at the relevant HR folder. Use a `label:` field in the config to set the
  display name (e.g. `name: Benchmark, label: DIV2K-Val`) — the registry key stays
  `Benchmark` while the results table shows the label.
- **RealSR** (`src/datasets/realsr.py`): paired *real* LR/HR folders (LR is not
  synthesized). Used for stage-2 fine-tuning and real-world evaluation.
- **Two stages** (per the project plan): stage 1 pretrains on synthetic DF2K pairs;
  stage 2 fine-tunes on RealSR via `--pretrained <ckpt>` with a smaller LR. DIV2K-Val
  is evaluation-only — never put it in `train_dataset`.
- **Losses** (`src/losses.py`, feed-forward only): `loss: l1|l2|charbonnier` plus
  optional `perceptual_weight > 0` (VGG relu2_2). Perceptual/GAN terms improve
  perceptual metrics but typically lower PSNR/SSIM.
- **val vs test metrics**: training validation uses the light `val_metrics`
  (default `[psnr, ssim]`); the full `metrics` list (incl. LPIPS + no-reference) is
  used by `evaluate.py`, since MUSIQ/CLIPIQA are slow and download weights.

## Diffusion model (SR3)

Core lives in `src/models/diffusion/`:
- `unet.py` — conditional U-Net noise predictor (timestep-embedded ResBlocks +
  attention at the deepest `attn_levels`). The condition is channel-concatenated to
  the noisy input, so `in_channels = target_channels + cond_channels`. Spatial sizes
  must be divisible by `2**(len(channel_mults)-1)`; `pad_to_multiple` handles
  arbitrary eval image sizes (pad → sample → crop).
- `gaussian_diffusion.py` — DDPM forward process + **DDIM** sampling. Diffusion runs
  in `[-1, 1]`; sampling cost is `sampling_timesteps` network calls (config knob).

**SR3** (`models/sr3.py`): diffuses the HR image; condition = `up(LR)` concatenated
to the noisy input.

Because sampling is expensive, the SR3 config uses a large `val_every` and a modest
`sampling_timesteps` (default 100), and evaluates DIV2K-Val on 256×256 crops
(`max_hr: 256`) — full 2K images are infeasible for the UNet self-attention.

## How a run is wired

A single YAML config (`configs/*.yaml`) fully specifies an experiment: `model`,
`train` hyperparameters, `train_dataset`, `val_datasets` (used during training),
`test_datasets` (used by `evaluate.py`), and `metrics`. `scripts/train.py` and
`scripts/evaluate.py` just parse the config, call the `build_*` helpers, and hand
off to `src/engine/`:

- `engine/trainer.py` — training loop (Adam + StepLR, periodic validation via the
  evaluator). Dispatches the loss on the model (diffusion `compute_loss` vs pixel
  loss). Saves `best.pth` when the monitored val metric improves and `last.pth` at
  the end; **early stopping** (`early_stop_patience`/`early_stop_metric`) halts when
  the metric stalls, so configs set a high `epochs` cap.
- `engine/evaluator.py` — runs a model over each test dataset, averages each metric,
  returns `{dataset: {metric: value}}` (the core comparison artifact).

## Known gaps / when extending

- The diffusion core is a faithful but compact reimplementation of the SR3
  architecture (DDPM + conditional UNet). It is not a checkpoint-compatible port of
  any official repo, so don't expect to load their pretrained weights. **SwinIR is
  the deliberate exception**: `src/models/swinir_arch.py` vendors the official
  architecture verbatim (timm helpers inlined) precisely so the released
  checkpoints load with `strict=True` — don't refactor that file; adapt in
  `src/models/swinir.py` instead. `evaluate.py` unwraps both this repo's
  `{"model": ...}` checkpoints and official `{"params"|"params_ema": ...}` ones.
- Diffusion evaluation is slow (sampling). For full benchmark sweeps prefer GPU and
  a small `sampling_timesteps`; the engine has no EMA or mixed-precision yet, both of
  which materially help diffusion training if you add them.
