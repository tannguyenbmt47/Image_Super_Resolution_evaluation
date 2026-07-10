#!/usr/bin/env bash
# Official pretrained checkpoints for models evaluated with released weights
# (currently SwinIR; see configs/swinir_x4.yaml). Weights land in ./weights.
set -euo pipefail
ROOT="${1:-weights}"
mkdir -p "$ROOT"
cd "$ROOT"

dl() {  # url out
  echo ">> $2"
  curl -L -C - --retry 8 --retry-delay 5 --fail -o "$2" "$1"
}

# SwinIR-M classical SR x4, trained on DF2K (Apache 2.0, JingyunLiang/SwinIR)
SWINIR_BASE="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"
f="001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
[ -f "$f" ] || dl "$SWINIR_BASE/$f" "$f"

echo "ALL WEIGHTS DOWNLOADED -> $ROOT/"
