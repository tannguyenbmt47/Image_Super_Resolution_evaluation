#!/usr/bin/env bash
# Datasets per the project plan:
#   pretrain : DF2K = DIV2K_train_HR + Flickr2K_HR   (synthetic LR-HR pairs)
#   evaluate : DIV2K_valid_HR (DIV2K-Val) + RealSR    (RealSR is manual, see end)
set -euo pipefail
ROOT="${1:-data}"
mkdir -p "$ROOT/DIV2K"
cd "$ROOT"

dl() {  # url out
  echo ">> $2"
  curl -L -C - --retry 8 --retry-delay 5 --fail -o "$2" "$1"
}

# DIV2K train HR (pretrain half of DF2K)
if [ ! -d DIV2K/DIV2K_train_HR ]; then
  dl "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip" DIV2K_train_HR.zip
  unzip -q -o DIV2K_train_HR.zip -d DIV2K/ && rm -f DIV2K_train_HR.zip
fi

# DIV2K valid HR (DIV2K-Val evaluation)
if [ ! -d DIV2K/DIV2K_valid_HR ]; then
  dl "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip" DIV2K_valid_HR.zip
  unzip -q -o DIV2K_valid_HR.zip -d DIV2K/ && rm -f DIV2K_valid_HR.zip
fi

# Flickr2K HR (the +Flickr2K half of DF2K, ~20GB). NOTE: SNU needs HTTP, not HTTPS.
if [ ! -d Flickr2K/Flickr2K_HR ]; then
  dl "http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar" Flickr2K.tar
  tar -xf Flickr2K.tar && rm -f Flickr2K.tar
fi

echo "ALL AUTO DOWNLOADS DONE"
echo "RealSR is not auto-downloaded (no stable public direct link)."
echo "Get it from https://github.com/csjcai/RealSR and arrange as:"
echo "  data/RealSR/{Train,Test}/{LR,HR}/   (matching filenames; see src/datasets/realsr.py)"
