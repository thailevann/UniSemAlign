#!/bin/bash
set -e

# Run from the UniSemAlign project root (script is in tools/).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

CONFIG="configs/crag_10.yaml"
LABELED="partitions/crag_10/labeled.txt"
UNLABELED="partitions/crag_10/unlabeled.txt"
SAVE="exp/crag/10/corrmatch"

mkdir -p "$SAVE"

python UniSemAlign.py \
  --config "$CONFIG" \
  --labeled-id-path "$LABELED" \
  --unlabeled-id-path "$UNLABELED" \
  --save-path "$SAVE" \
  2>&1 | tee "$SAVE/train.log"
