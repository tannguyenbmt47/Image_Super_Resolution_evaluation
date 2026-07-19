#!/usr/bin/env bash
# Train 4 models sequentially, same 20-epoch budget. Logs: train_<model>.log
set -u
cd "$(dirname "$0")"
for m in srcnn srgan swinir sr3; do
  echo "===== $(date '+%H:%M:%S') TRAIN $m ====="
  .venv/bin/python scripts/train.py --config "configs/${m}_df2k_x4.yaml" \
    > "train_${m}.log" 2>&1
  echo "===== $(date '+%H:%M:%S') DONE $m (exit $?) ====="
done
echo "ALL DONE $(date '+%H:%M:%S')"
