#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="$SCRIPT_DIR/run_evaluation.sh"
RESULTS_BASE="$SCRIPT_DIR/results/lora_retriever_eval"

MODEL="qwen2b"
GPU_ID="2"
APPS=(amazon clock ebay esty flipkart google_drive reminder youtube)
METHODS=(fusion mixture)

# bash evaluation/run_eval_id.sh --model qwen2b --gpu_id 0
usage() {
  cat <<USAGE
Usage:
  bash evaluation/run_eval_id.sh [options]

Options:
  --model MODEL        qwen2b|qwen7b|intern2b (default: qwen2b)
  --gpu_id ID          GPU id (default: 0)
  --gou_id ID          Alias of --gpu_id
  --apps "a b c d"     Space-separated app names
  --methods "m1 m2"    Space-separated merge methods (default: "fusion mixture")
                       allowed: fusion mixture

Example:
  bash evaluation/run_eval_id.sh --model qwen2b --gpu_id 0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --gpu_id|--gou_id) GPU_ID="$2"; shift 2 ;;
    --apps)
      IFS=' ' read -r -a APPS <<< "$2"
      shift 2
      ;;
    --methods)
      IFS=' ' read -r -a METHODS <<< "$2"
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1"; usage; exit 1 ;;
  esac
done

normalize_app() {
  local app="$1"
  case "$app" in
    remider) echo "reminder" ;;
    *) echo "$app" ;;
  esac
}

extract_percent() {
  local pattern="$1"
  local file="$2"
  local v
  v="$(grep -E "$pattern" "$file" | sed -E 's/.*: *([0-9.]+%).*/\1/' | tail -n1 || true)"
  if [[ -z "$v" ]]; then
    echo "N/A"
  else
    echo "$v"
  fi
}

if [[ ! -f "$RUN_ONE" ]]; then
  echo "[ERROR] run script not found: $RUN_ONE"
  exit 1
fi
mkdir -p "$RESULTS_BASE"

TS="$(date +"%Y%m%d_%H%M%S")"
SUMMARY_FILE="$RESULTS_BASE/summary_4apps_${TS}.tsv"

for method in "${METHODS[@]}"; do
  case "$method" in
    fusion|mixture) ;;
    *)
      echo "[ERROR] Unsupported method: $method (allowed: fusion mixture)"
      exit 1
      ;;
  esac
done

printf "app\tmethod\tstep_accuracy\tepisode_accuracy\trun_dir\n" > "$SUMMARY_FILE"

echo "============================================================"
echo "Batch eval starts"
echo "model=$MODEL, gpu_id=$GPU_ID"
echo "apps=${APPS[*]}"
echo "methods=${METHODS[*]}"
echo "============================================================"

for raw_app in "${APPS[@]}"; do
  app="$(normalize_app "$raw_app")"
  if [[ "$raw_app" != "$app" ]]; then
    echo "[INFO] app alias mapped: $raw_app -> $app"
  fi

  for method in "${METHODS[@]}"; do
    echo
    echo "[RUN] app=$app method=$method"
    method_output_base="$RESULTS_BASE/$method"
    bash "$RUN_ONE" \
      --test_input "$app" \
      --model "$MODEL" \
      --gou_id "$GPU_ID" \
      --merge_method "$method" \
      --output_base "$method_output_base"

    dataset_dir="$method_output_base/${app}_train"
    if [[ ! -d "$dataset_dir" ]]; then
      echo "[ERROR] result dir not found: $dataset_dir"
      exit 1
    fi

    run_dir="$(ls -1dt "$dataset_dir"/* | head -n1)"
    step_log="$run_dir/eval_step.log"
    episode_log="$run_dir/eval_episode.log"

    step_acc="$(extract_percent "Step-level Accuracy" "$step_log")"
    episode_acc="$(extract_percent "Episode-level Accuracy" "$episode_log")"

    printf "%s\t%s\t%s\t%s\t%s\n" "$app" "$method" "$step_acc" "$episode_acc" "$run_dir" >> "$SUMMARY_FILE"
    echo "[DONE] $app [$method] | step=$step_acc | episode=$episode_acc"
  done
done

echo

echo "==================== Summary ===================="
column -t -s $'\t' "$SUMMARY_FILE" || cat "$SUMMARY_FILE"
echo "================================================="
echo "Saved summary: $SUMMARY_FILE"
