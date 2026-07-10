#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

RUN_MODE="${RUN_MODE:-quick}"
DATASETS="${DATASETS:-shs27k shs148k}"
SPLITS="${SPLITS:-random bfs dfs}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-1024}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/save_model}"
METRICS_CSV="${METRICS_CSV:-./outputs/proposal_results.csv}"
INTERACTIVE_UI="${INTERACTIVE_UI:-True}"
SPLIT_NEW="${SPLIT_NEW:-auto}"
RESET_CSV="${RESET_CSV:-True}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x ".venv310/bin/python" ]]; then
    PYTHON=".venv310/bin/python"
  else
    PYTHON="python"
  fi
fi

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_proposal_experiments.sh [quick|full] [options]

Modes:
  quick                 Run all proposal ablations on SHS27k/random only.
  full                  Run all proposal ablations on all configured datasets/splits.

Options:
  --epochs N            Training epochs. Default: 400
  --batch-size N        Batch size. Default: 1024
  --test-batch-size N   Evaluation edge batch size. Default: 1024
  --metrics-csv PATH    CSV output path. Default: ./outputs/proposal_results.csv
  --output-dir PATH     Checkpoint output dir. Default: ./outputs/save_model
  --datasets "LIST"     Dataset list for full mode. Default: "shs27k shs148k"
  --splits "LIST"       Split list for full mode. Default: "random bfs dfs"
  --split-new BOOL      Regenerate split json files: True, False, or auto. Default: auto
  --interactive-ui BOOL Enable train dashboard. Default: True
  --python PATH        Python interpreter. Default: .venv310/bin/python when present
  --no-reset-csv        Append to CSV instead of recreating it.
  -h, --help            Show this help.

Examples:
  bash scripts/run_proposal_experiments.sh
  bash scripts/run_proposal_experiments.sh quick --epochs 20
  bash scripts/run_proposal_experiments.sh full --datasets "shs27k" --splits "random bfs dfs"
EOF
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  RUN_MODE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --test-batch-size)
      TEST_BATCH_SIZE="$2"
      shift 2
      ;;
    --metrics-csv)
      METRICS_CSV="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --splits)
      SPLITS="$2"
      shift 2
      ;;
    --split-new)
      SPLIT_NEW="$2"
      shift 2
      ;;
    --interactive-ui)
      INTERACTIVE_UI="$2"
      shift 2
      ;;
    --python)
      PYTHON="$2"
      shift 2
      ;;
    --no-reset-csv)
      RESET_CSV="False"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

preflight() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter is not executable: $PYTHON" >&2
    exit 1
  fi

  "$PYTHON" - <<'PY'
import importlib
import os
import sys

missing = []
for module in ("numpy", "torch", "torch_geometric", "torch_scatter", "tqdm"):
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append("{} ({})".format(module, exc))

required_paths = [
    "assets/data/protein.actions.SHS27k.STRING.txt",
    "assets/data/protein.SHS27k.sequences.dictionary.tsv",
    "assets/data/vec5_CTC.txt",
    "assets/pretrained/shs_MASSA.pickle",
]
missing_paths = [path for path in required_paths if not os.path.exists(path)]

if missing or missing_paths:
    if missing:
        print("Missing Python dependencies:", file=sys.stderr)
        for item in missing:
            print("  - {}".format(item), file=sys.stderr)
    if missing_paths:
        print("Missing required input files:", file=sys.stderr)
        for path in missing_paths:
            print("  - {}".format(path), file=sys.stderr)
    sys.exit(1)
PY
}

split_new_for() {
  local dataset="$1"
  local split="$2"
  if [[ "$SPLIT_NEW" != "auto" ]]; then
    echo "$SPLIT_NEW"
    return
  fi

  local index_path="assets/splits_${dataset}/${dataset}.${split}.fold1.json"
  if [[ -f "$index_path" ]]; then
    echo "False"
  else
    echo "True"
  fi
}

preflight

if [[ "$RESET_CSV" == "True" ]]; then
  rm -f "$METRICS_CSV"
fi
"$PYTHON" -m src.train.seed_paper_results --output_csv "$METRICS_CSV"

run_train() {
  local dataset="$1"
  local split="$2"
  local description="$3"
  local feature_source="$4"
  local fusion_strategy="$5"
  local loss_type="$6"
  local subgraph_hops="$7"
  local split_new
  split_new="$(split_new_for "$dataset" "$split")"

  echo "Running ${description}: dataset=${dataset}, split=${split}, split_new=${split_new}"
  "$PYTHON" main.py \
    --dataset_type "$dataset" \
    --mode "$split" \
    --split_mode "$split" \
    --split_new "$split_new" \
    --description "$description" \
    --feature_source "$feature_source" \
    --fusion_strategy "$fusion_strategy" \
    --loss_type "$loss_type" \
    --subgraph_hops "$subgraph_hops" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --metrics_csv "$METRICS_CSV" \
    --interactive_ui "$INTERACTIVE_UI"

  local checkpoint
  checkpoint="$("$PYTHON" - <<PY
import glob
import os

pattern = os.path.join("$OUTPUT_DIR", "${split}_${dataset}", "gnn_${description}_*", "gnn_model_valid_best.ckpt")
candidates = glob.glob(pattern)
print(max(candidates, key=os.path.getmtime) if candidates else "")
PY
)"
  if [[ -z "$checkpoint" ]]; then
    echo "Could not find best checkpoint for ${description}" >&2
    exit 1
  fi

  echo "Testing ${description}: checkpoint=${checkpoint}"
  "$PYTHON" main_test.py \
    --dataset_type "$dataset" \
    --mode "$split" \
    --description "$description" \
    --gnn_model "$checkpoint" \
    --feature_source "$feature_source" \
    --fusion_strategy "$fusion_strategy" \
    --loss_type "$loss_type" \
    --subgraph_hops "$subgraph_hops" \
    --test_batch_size "$TEST_BATCH_SIZE" \
    --metrics_csv "$METRICS_CSV" \
    --test_all True
}

if [[ "$RUN_MODE" == "quick" ]]; then
  run_train "shs27k" "random" "paper_fixed_sum_baseline" "both" "fixed_sum" "asl" "1"
  run_train "shs27k" "random" "proposal_global_only" "global" "fixed_sum" "asl" "1"
  run_train "shs27k" "random" "proposal_local_only" "local" "fixed_sum" "asl" "1"
  run_train "shs27k" "random" "proposal_concat_mlp_fusion" "both" "concat_mlp" "asl" "1"
  run_train "shs27k" "random" "proposal_scalar_gate_fusion" "both" "dynamic" "asl" "1"
  run_train "shs27k" "random" "proposal_feature_wise_adaptive" "both" "feature_wise" "asl" "1"
  run_train "shs27k" "random" "proposal_bce_loss" "both" "fixed_sum" "bce" "1"
  run_train "shs27k" "random" "proposal_k_hop_2" "both" "fixed_sum" "asl" "2"
elif [[ "$RUN_MODE" == "full" ]]; then
  for dataset in $DATASETS; do
    for split in $SPLITS; do
      run_train "$dataset" "$split" "global_only" "global" "fixed_sum" "asl" "1"
      run_train "$dataset" "$split" "local_only" "local" "fixed_sum" "asl" "1"

      run_train "$dataset" "$split" "fixed_sum" "both" "fixed_sum" "asl" "1"
      run_train "$dataset" "$split" "concat_mlp" "both" "concat_mlp" "asl" "1"
      run_train "$dataset" "$split" "scalar_gate" "both" "dynamic" "asl" "1"
      run_train "$dataset" "$split" "feature_wise_adaptive" "both" "feature_wise" "asl" "1"

      run_train "$dataset" "$split" "bce_loss" "both" "fixed_sum" "bce" "1"
      run_train "$dataset" "$split" "asl_loss" "both" "fixed_sum" "asl" "1"
    done
  done

  for split in $SPLITS; do
    run_train "shs27k" "$split" "k_hop_1" "both" "feature_wise" "asl" "1"
    run_train "shs27k" "$split" "k_hop_2" "both" "feature_wise" "asl" "2"
  done
else
  echo "Unknown RUN_MODE=$RUN_MODE. Use quick or full." >&2
  exit 1
fi

echo "Done. Results CSV: $METRICS_CSV"
