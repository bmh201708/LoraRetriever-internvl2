#!/bin/bash
#
# LoraRetriever Evaluation Scripts
# Run different evaluation scenarios
#

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Lretriever

# Set environment variables
export MAX_PIXELS=150000
export MAX_NUM=12
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
TOP_K=${TOP_K:-3}
MERGE_METHOD=${MERGE_METHOD:-mixture}
THRESHOLD=${THRESHOLD:-0.6}
TEST_DATA=${TEST_DATA:-data/Val_100.jsonl}
EVAL_TYPE=${EVAL_TYPE:-all}
MAX_SAMPLES=${MAX_SAMPLES:-}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --merge_method)
            MERGE_METHOD="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --eval_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --evaluate_only)
            EVALUATE_ONLY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "LoraRetriever Evaluation"
echo "=============================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Top-K: $TOP_K"
echo "Merge Method: $MERGE_METHOD"
echo "Threshold: $THRESHOLD"
echo "Test Data: $TEST_DATA"
echo "Eval Type: $EVAL_TYPE"
echo "=============================================="

# Evaluation only mode
if [ -n "$EVALUATE_ONLY" ]; then
    echo "Running evaluation only on: $EVALUATE_ONLY"
    python evaluation/evaluate.py \
        --data_path "$EVALUATE_ONLY" \
        --threshold "$THRESHOLD" \
        --verbose
    exit 0
fi

# Run full evaluation pipeline
CMD="python evaluation/run_evaluation.py \
    --test_data \"$TEST_DATA\" \
    --top_k $TOP_K \
    --merge_method $MERGE_METHOD \
    --threshold $THRESHOLD \
    --eval_type $EVAL_TYPE"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

echo "Running: $CMD"
eval $CMD

echo ""
echo "=============================================="
echo "Evaluation Complete"
echo "=============================================="
