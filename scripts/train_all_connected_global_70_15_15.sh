#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv310/bin/python" ]]; then
    PYTHON=".venv310/bin/python"
  else
    PYTHON="python"
  fi
fi

EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-262144}"
DEVICE="${DEVICE:-auto}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/save_model}"
METRICS_CSV="${METRICS_CSV:-./outputs/all_connected_global_70_15_15.csv}"
INDEX_PATH="${INDEX_PATH:-./assets/splits_string/string.random.70_15_15.json}"
INTERACTIVE_UI="${INTERACTIVE_UI:-False}"
SPLIT_NEW="${SPLIT_NEW:-auto}"

usage() {
  cat <<'EOF'
Train the all_connected dataset with the global branch only and a random
70% train / 15% validation / 15% independent test split.

Usage:
  bash scripts/train_all_connected_global_70_15_15.sh

Environment overrides:
  PYTHON, EPOCHS, BATCH_SIZE, DEVICE, CHECKPOINT_INTERVAL, OUTPUT_DIR,
  METRICS_CSV, INDEX_PATH, INTERACTIVE_UI, SPLIT_NEW

Defaults:
  EPOCHS=400, BATCH_SIZE=262144, DEVICE=auto, INTERACTIVE_UI=False
  SPLIT_NEW=auto creates the split once and reuses it on later runs.

Example:
  DEVICE=cuda EPOCHS=400 BATCH_SIZE=262144 \
    bash scripts/train_all_connected_global_70_15_15.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  usage >&2
  exit 1
fi

if ! command -v "$PYTHON" >/dev/null 2>&1 && [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter is unavailable: $PYTHON" >&2
  exit 1
fi

for path in \
  assets/data/9606.protein.actions.all_connected.txt \
  assets/data/protein.STRING_all_connected.sequences.dictionary.tsv \
  assets/data/vec5_CTC.txt; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

case "$SPLIT_NEW" in
  auto)
    if [[ -f "$INDEX_PATH" ]]; then
      split_option="--no-split-new"
    else
      split_option="--split-new"
    fi
    ;;
  True|true)
    split_option="--split-new"
    ;;
  False|false)
    split_option="--no-split-new"
    ;;
  *)
    echo "SPLIT_NEW must be auto, True, or False; got: $SPLIT_NEW" >&2
    exit 1
    ;;
esac

case "$INTERACTIVE_UI" in
  True|true)
    ui_option="--interactive-ui"
    ;;
  False|false)
    ui_option="--no-interactive-ui"
    ;;
  *)
    echo "INTERACTIVE_UI must be True or False; got: $INTERACTIVE_UI" >&2
    exit 1
    ;;
esac

echo "Training global-only all_connected with train:validation:test=70:15:15"
echo "Split index: $INDEX_PATH"
echo "Epochs: $EPOCHS, batch size: $BATCH_SIZE, device: $DEVICE, split option: $split_option"

"$PYTHON" main.py \
  --dataset_type string \
  --mode random \
  --split_mode random \
  --validation_size 0.15 \
  --test_size 0.15 \
  --train_valid_index_path "$INDEX_PATH" \
  "$split_option" \
  --description global_only_all_connected_70_15_15 \
  --feature_source global \
  --local_encoder sparse \
  --fusion_strategy fixed_sum \
  --loss_type asl \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL" \
  --output_dir "$OUTPUT_DIR" \
  --metrics_csv "$METRICS_CSV" \
  "$ui_option"
