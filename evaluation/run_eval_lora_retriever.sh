#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INFER_SCRIPT="$REPO_ROOT/infer_lora_retriever.py"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_inference.py"

TEST_INPUT=""
OUTPUT_BASE="$SCRIPT_DIR/results/lora_retriever_eval"
DATA_ROOT=""
DATA_ROOT_SET="0"
DEFAULT_APP_DATA_ROOT="/home/hmpiao/hmpiao/jinyike/FedMABench/data_new_test/app"
DEFAULT_CATEGORY_DATA_ROOT="/home/hmpiao/hmpiao/jinyike/FedMABench/data_new_test/category"

MODEL_ALIAS="qwen2b"
MODEL_TYPE=""
MODEL_PATH=""
APP_CONFIG=""
CATEGORY_CONFIG=""
LORA_TYPE="all"
MERGE_METHOD="fusion"
COMPOSITION_WEIGHT_MODE="uniform"
TOP_K="3"
GPU_ID="0"
MAX_NEW_TOKENS="512"
TEMPERATURE="0.0"
NUM_SAMPLES=""

# example: bash evaluation/run_eval_lora_retriever.sh --test_input amazon --model qwen2b --merge_method mixture --gpu_id 6
usage() {
  cat <<USAGE
Usage:
  bash evaluation/run_eval_lora_retriever.sh --test_input <app_name> --model <qwen2b|qwen7b|intern2b> [options]

Required:
  --test_input APP             App name, e.g. amazon
  --model MODEL                qwen2b | qwen7b | intern2b

Optional:
  --data_root DIR              Dataset root (default by --lora_type: app->.../app, category->.../category)
  --output_base DIR            Output base directory
  --model_path PATH            Override base model path
  --app_config PATH            Override app LoRA config
  --category_config PATH       Override category LoRA config
  --lora_type TYPE             all|app|category (default: all)
  --merge_method METHOD        mixture|fusion (default: fusion)
  --composition_weight_mode M  uniform|weighted (default: uniform)
  --top_k N                    Default: 3
  --gpu_id ID                  Default: 0
  --max_new_tokens N           Default: 512
  --temperature FLOAT          Default: 0.0
  --num_samples N              Optional debug subset

Example:
  bash evaluation/run_eval_lora_retriever.sh \
    --test_input amazon \
    --model qwen2b \
    --merge_method fusion \
    --composition_weight_mode uniform \
    --top_k 3 \
    --gpu_id 0
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test_input) TEST_INPUT="$2"; shift 2 ;;
    --model) MODEL_ALIAS="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; DATA_ROOT_SET="1"; shift 2 ;;
    --output_base) OUTPUT_BASE="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --app_config) APP_CONFIG="$2"; shift 2 ;;
    --category_config) CATEGORY_CONFIG="$2"; shift 2 ;;
    --lora_type) LORA_TYPE="$2"; shift 2 ;;
    --merge_method) MERGE_METHOD="$2"; shift 2 ;;
    --composition_weight_mode) COMPOSITION_WEIGHT_MODE="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$TEST_INPUT" ]]; then
  echo "[ERROR] --test_input is required"
  usage
  exit 1
fi
if [[ -z "$MODEL_ALIAS" ]]; then
  echo "[ERROR] --model is required"
  usage
  exit 1
fi

case "$MODEL_ALIAS" in
  qwen2b) MODEL_TYPE="qwen2-vl-2b-instruct" ;;
  qwen7b) MODEL_TYPE="qwen2-vl-7b-instruct" ;;
  intern2b) MODEL_TYPE="internvl2-2b" ;;
  *)
    echo "[ERROR] Unsupported --model: $MODEL_ALIAS (expected: qwen2b|qwen7b|intern2b)"
    exit 1
    ;;
esac

case "$LORA_TYPE" in
  all|app|category) ;;
  *)
    echo "[ERROR] Unsupported --lora_type: $LORA_TYPE (expected: all|app|category)"
    exit 1
    ;;
esac

if [[ "$DATA_ROOT_SET" != "1" ]]; then
  case "$LORA_TYPE" in
    category) DATA_ROOT="$DEFAULT_CATEGORY_DATA_ROOT" ;;
    *) DATA_ROOT="$DEFAULT_APP_DATA_ROOT" ;;
  esac
fi

# Set default app config by model alias when user does not provide --app_config.
if [[ -z "$APP_CONFIG" && "$LORA_TYPE" != "category" ]]; then
  case "$MODEL_ALIAS" in
    qwen2b)
      APP_CONFIG="/home/hmpiao/hmpiao/jinyike/LoraRetriever-internvl2/config/app_loras_config_qwen2vl.json"
      ;;
    qwen7b)
      APP_CONFIG="/data1/hmpiao/jinyike/LoraRetriever-internvl2/config/app_loras_config_qwen7b.json"
      ;;
  esac
fi

# Set default category config by model alias when user does not provide --category_config.
if [[ -z "$CATEGORY_CONFIG" && "$LORA_TYPE" != "app" ]]; then
  case "$MODEL_ALIAS" in
    qwen2b|qwen7b)
      CATEGORY_CONFIG="/data1/hmpiao/jinyike/LoraRetriever-internvl2/config/category_loras_config_qwen2vl.json"
      ;;
    intern2b)
      CATEGORY_CONFIG="/data1/hmpiao/jinyike/LoraRetriever-internvl2/config/category_loras_config_internvl2.json"
      ;;
  esac
fi

# Hard gate configs by lora_type so wrong type cannot be mixed in.
if [[ "$LORA_TYPE" == "app" ]]; then
  CATEGORY_CONFIG="/dev/null"
fi
if [[ "$LORA_TYPE" == "category" ]]; then
  APP_CONFIG="/dev/null"
fi

if [[ ! -f "$INFER_SCRIPT" ]]; then
  echo "[ERROR] infer script not found: $INFER_SCRIPT"
  exit 1
fi
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] eval script not found: $EVAL_SCRIPT"
  exit 1
fi
if [[ "$TEST_INPUT" == *.jsonl || "$TEST_INPUT" == */* ]]; then
  DATASET_PATH="$TEST_INPUT"
  APP_NAME="$(basename "${TEST_INPUT%.*}")"
else
  APP_NAME="$TEST_INPUT"
  DATASET_PATH="$DATA_ROOT/${APP_NAME}_train.jsonl"
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[ERROR] dataset not found: $DATASET_PATH"
  exit 1
fi

# Strict dataset type checks.
if [[ "$LORA_TYPE" == "app" ]]; then
  if [[ "$DATASET_PATH" == *"/category/"* || "$DATA_ROOT" == *"/category"* ]]; then
    echo "[ERROR] --lora_type app requires app test set, but got: $DATASET_PATH"
    exit 1
  fi
  if [[ "$DATASET_PATH" != *"/app/"* ]]; then
    echo "[ERROR] --lora_type app requires dataset path under .../app/, got: $DATASET_PATH"
    exit 1
  fi
fi
if [[ "$LORA_TYPE" == "category" ]]; then
  if [[ "$DATASET_PATH" == *"/app/"* || "$DATA_ROOT" == *"/app"* ]]; then
    echo "[ERROR] --lora_type category requires category test set, but got: $DATASET_PATH"
    exit 1
  fi
  if [[ "$DATASET_PATH" != *"/category/"* ]]; then
    echo "[ERROR] --lora_type category requires dataset path under .../category/, got: $DATASET_PATH"
    exit 1
  fi
fi

# Strict config existence checks for selected lora_type.
if [[ "$LORA_TYPE" == "app" ]]; then
  if [[ "$APP_CONFIG" == "/dev/null" || ! -f "$APP_CONFIG" ]]; then
    echo "[ERROR] --lora_type app requires a valid app config, got: $APP_CONFIG"
    exit 1
  fi
fi
if [[ "$LORA_TYPE" == "category" ]]; then
  if [[ "$CATEGORY_CONFIG" == "/dev/null" || ! -f "$CATEGORY_CONFIG" ]]; then
    echo "[ERROR] --lora_type category requires a valid category config, got: $CATEGORY_CONFIG"
    exit 1
  fi
fi

mkdir -p "$OUTPUT_BASE"

# Avoid cache/permission warnings in shared environments.
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl_cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/tmp/modelscope_cache}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"

build_common_args() {
  COMMON_ARGS=(
    --model_type "$MODEL_TYPE"
    --test_data "$1"
    --lora_type "$LORA_TYPE"
    --merge_method "$MERGE_METHOD"
    --composition_weight_mode "$COMPOSITION_WEIGHT_MODE"
    --top_k "$TOP_K"
    --gpu_id "$GPU_ID"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
  )

  if [[ -n "$MODEL_PATH" ]]; then
    COMMON_ARGS+=(--model_path "$MODEL_PATH")
  fi
  if [[ -n "$APP_CONFIG" ]]; then
    COMMON_ARGS+=(--app_config "$APP_CONFIG")
  fi
  if [[ -n "$CATEGORY_CONFIG" ]]; then
    COMMON_ARGS+=(--category_config "$CATEGORY_CONFIG")
  fi
  if [[ -n "$NUM_SAMPLES" ]]; then
    COMMON_ARGS+=(--num_samples "$NUM_SAMPLES")
  fi
}

run_one_dataset() {
  local dataset_path="$1"
  local dataset_name
  dataset_name="$(basename "${dataset_path%.*}")"

  local run_id
  run_id="$(date +"%Y%m%d_%H%M%S")"
  local run_dir="$OUTPUT_BASE/$dataset_name/$run_id"
  local step_out_dir="$run_dir/step_outputs"
  local episode_out_dir="$run_dir/episode_outputs"

  mkdir -p "$step_out_dir" "$episode_out_dir"

  echo "============================================================"
  echo "Dataset    : $dataset_path"
  echo "Run Dir    : $run_dir"
  echo "Model      : $MODEL_ALIAS -> $MODEL_TYPE"
  echo "LoRA Type  : $LORA_TYPE"
  echo "App Config : ${APP_CONFIG:-<none>}"
  echo "Cat Config : ${CATEGORY_CONFIG:-<none>}"
  echo "Merge      : $MERGE_METHOD"
  echo "WeightMode : $COMPOSITION_WEIGHT_MODE"
  echo "Top-K      : $TOP_K"
  echo "GPU        : $GPU_ID"
  echo "============================================================"

  build_common_args "$dataset_path"

  echo
  echo "[1/4] Infer STEP mode (history uses ground truth)"
  python "$INFER_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --dialog_mode step \
    --output_dir "$step_out_dir" \
    2>&1 | tee "$run_dir/infer_step.log"

  local step_result
  step_result="$(ls -1t "$step_out_dir"/*.jsonl | head -n1)"
  if [[ -z "$step_result" ]]; then
    echo "[ERROR] STEP result file not found in $step_out_dir"
    exit 1
  fi

  echo
  echo "[2/4] Infer EPISODE mode (history uses model predictions)"
  python "$INFER_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --dialog_mode episode \
    --output_dir "$episode_out_dir" \
    2>&1 | tee "$run_dir/infer_episode.log"

  local episode_result
  episode_result="$(ls -1t "$episode_out_dir"/*.jsonl | head -n1)"
  if [[ -z "$episode_result" ]]; then
    echo "[ERROR] EPISODE result file not found in $episode_out_dir"
    exit 1
  fi

  echo
  echo "[3/4] Evaluate STEP result (expect step accuracy output)"
  python "$EVAL_SCRIPT" --results_path "$step_result" \
    2>&1 | tee "$run_dir/eval_step.log"

  echo
  echo "[4/4] Evaluate EPISODE result (expect episode accuracy output)"
  python "$EVAL_SCRIPT" --results_path "$episode_result" \
    2>&1 | tee "$run_dir/eval_episode.log"

  cat > "$run_dir/summary_paths.txt" <<SUMMARY
DATASET=$dataset_path
STEP_RESULT=$step_result
EPISODE_RESULT=$episode_result
STEP_EVAL_LOG=$run_dir/eval_step.log
EPISODE_EVAL_LOG=$run_dir/eval_episode.log
SUMMARY

  echo
  echo "[DONE] $dataset_name"
  echo "  STEP result   : $step_result"
  echo "  EPISODE result: $episode_result"
  echo "  Eval logs     : $run_dir/eval_step.log, $run_dir/eval_episode.log"
}

run_one_dataset "$DATASET_PATH"
