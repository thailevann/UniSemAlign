#!/bin/bash
set -e

# Run from the UniSemAlign project root (script is in tools/).
# Data: image .tif, mask .png. data_root in the config should point to the dataset (contains glas/10/...).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

CONFIG="configs/glas_10.yaml"
LABELED="partitions/glas_10/labeled.txt"
UNLABELED="partitions/glas_10/unlabeled.txt"
SAVE="exp/glas/10/unisemalign"

mkdir -p "$SAVE"

python UniSemAlign.py \
  --config "$CONFIG" \
  --labeled-id-path "$LABELED" \
  --unlabeled-id-path "$UNLABELED" \
  --save-path "$SAVE" \
  2>&1 | tee "$SAVE/train.log"
