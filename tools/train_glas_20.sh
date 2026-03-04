#!/bin/bash
set -e

# Run from the UniSemAlign project root (script is in tools/).
# Data: image .tif, mask .png. data_root in config should point to the dataset (contains glas/20/...).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

CONFIG="configs/glas_20.yaml"
LABELED="partitions/glas_20/labeled.txt"
UNLABELED="partitions/glas_20/unlabeled.txt"
SAVE="exp/glas/20/corrmatch"

mkdir -p "$SAVE"

python UniSemAlign.py \
  --config "$CONFIG" \
  --labeled-id-path "$LABELED" \
  --unlabeled-id-path "$UNLABELED" \
  --save-path "$SAVE" \
  2>&1 | tee "$SAVE/train.log"
