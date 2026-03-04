
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

CHECKPOINT="${1:?Usage: $0 <checkpoint_path> [split] [out_dir] [mode=score|mask|both] [port]}"
SPLIT="${2:-val}"
OUT_DIR="${3:-visual}"
MODE="${4:-both}"
PORT="${5:-29501}"

if [[ "$CHECKPOINT" != /* ]]; then
  CHECKPOINT_ABS="$REPO/$CHECKPOINT"
else
  CHECKPOINT_ABS="$CHECKPOINT"
fi

if [[ ! -f "$CHECKPOINT_ABS" ]]; then
  echo "Checkpoint not found: $CHECKPOINT_ABS"
  exit 1
fi

echo "Config: configs/glas_10.yaml"
echo "Checkpoint: $CHECKPOINT_ABS"
echo "Split: $SPLIT"
echo "Out dir: $OUT_DIR"
echo "Mode: $MODE"
echo ""

torchrun --nproc_per_node=1 --master_port="$PORT" gen_masks.py \
  --config configs/glas_10.yaml \
  --checkpoint_path "$CHECKPOINT_ABS" \
  --split "$SPLIT" \
  --out-dir "$OUT_DIR" \
  --mode "$MODE" \
  --port "$PORT"
