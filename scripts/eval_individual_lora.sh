#!/bin/bash
# ======================================================================
# Individual LoRA Evaluation Script
# 
# 测评单个LoRA在其对应测试数据上的表现，不涉及LoRA检索和融合
# 支持 InternVL2-2B 和 Qwen2-VL-7B 两种模型
# 支持 App LoRA 和 Category LoRA 两种类型
#
# 用法:
#   bash scripts/eval_individual_lora.sh --model qwen2vl --lora_type app --lora_name adidas --gpu_id 0
#   bash scripts/eval_individual_lora.sh --model internvl2 --lora_type category --lora_name all --gpu_id 0
# ======================================================================

set -e

# 默认参数
MODEL="qwen2vl"           # qwen2vl or internvl2
LORA_TYPE="app"           # app or category
LORA_NAME="all"           # specific lora name or "all"
GPU_ID="0"
SAMPLES="-1"              # -1 表示完整测试集

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --lora_type)
            LORA_TYPE="$2"
            shift 2
            ;;
        --lora_name)
            LORA_NAME="$2"
            shift 2
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/eval_individual_lora.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model      Model type: qwen2vl or internvl2 (default: qwen2vl)"
            echo "  --lora_type  LoRA type: app or category (default: app)"
            echo "  --lora_name  LoRA name or 'all' to evaluate all (default: all)"
            echo "  --gpu_id     GPU ID to use (default: 0)"
            echo "  --samples    Number of samples to test, -1 for all (default: -1)"
            echo ""
            echo "Examples:"
            echo "  bash scripts/eval_individual_lora.sh --model qwen2vl --lora_type app --lora_name adidas --gpu_id 0"
            echo "  bash scripts/eval_individual_lora.sh --model internvl2 --lora_type category --lora_name all --gpu_id 0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 获取脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 设置模型配置
if [ "$MODEL" = "qwen2vl" ]; then
    MODEL_TYPE="qwen2-vl-7b-instruct"
    MODEL_PATH="/home/hmpiao/hmpiao/Qwen2-VL-7B-Instruct"
    if [ "$LORA_TYPE" = "app" ]; then
        CONFIG_FILE="$PROJECT_ROOT/config/app_loras_config_qwen2vl.json"
    else
        CONFIG_FILE="$PROJECT_ROOT/config/category_loras_config_qwen2vl.json"
    fi
elif [ "$MODEL" = "internvl2" ]; then
    MODEL_TYPE="internvl2-2b"
    MODEL_PATH="/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B"
    if [ "$LORA_TYPE" = "app" ]; then
        CONFIG_FILE="$PROJECT_ROOT/config/app_loras_config_internvl2.json"
    else
        CONFIG_FILE="$PROJECT_ROOT/config/category_loras_config_internvl2.json"
    fi
else
    echo "Error: Unknown model type '$MODEL'. Use 'qwen2vl' or 'internvl2'."
    exit 1
fi

# 设置测试数据目录
if [ "$LORA_TYPE" = "app" ]; then
    TEST_DATA_DIR="$PROJECT_ROOT/data/test_data_by_app"
elif [ "$LORA_TYPE" = "category" ]; then
    TEST_DATA_DIR="$PROJECT_ROOT/data/test_data_by_category"
else
    echo "Error: Unknown lora_type '$LORA_TYPE'. Use 'app' or 'category'."
    exit 1
fi

# 设置输出目录
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_BASE="$PROJECT_ROOT/output/individual_lora_eval/${MODEL}_${LORA_TYPE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE"

# 结果CSV文件
RESULTS_CSV="$OUTPUT_BASE/results_summary.csv"
echo "model,lora_type,lora_name,step_accuracy,episode_accuracy" > "$RESULTS_CSV"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=100000
export MAX_NUM=6

echo "========================================================================"
echo "Individual LoRA Evaluation"
echo "========================================================================"
echo "Model: $MODEL ($MODEL_TYPE)"
echo "Model Path: $MODEL_PATH"
echo "LoRA Type: $LORA_TYPE"
echo "LoRA Name: $LORA_NAME"
echo "GPU: $GPU_ID"
echo "Config: $CONFIG_FILE"
echo "Test Data Dir: $TEST_DATA_DIR"
echo "Output: $OUTPUT_BASE"
echo "========================================================================"

# 从配置文件中提取LoRA信息的Python脚本
get_lora_info() {
    python3 - "$CONFIG_FILE" "$1" <<'EOF'
import json
import sys

config_file = sys.argv[1]
lora_name_filter = sys.argv[2] if len(sys.argv) > 2 else "all"

with open(config_file, 'r') as f:
    loras = json.load(f)

for lora in loras:
    name = lora['lora_name']
    path = lora['lora_path']
    
    # 从lora_name中提取app/category名称
    # 格式: app_lora_xxx 或 category_lora_xxx 或 app_lora_xxx_qwen2vl
    parts = name.split('_')
    if 'qwen2vl' in parts:
        parts.remove('qwen2vl')
    # 提取实际名称 (跳过 app_lora_ 或 category_lora_ 前缀)
    if 'app' in parts and 'lora' in parts:
        idx = parts.index('lora')
        data_name = '_'.join(parts[idx+1:])
    elif 'category' in parts and 'lora' in parts:
        idx = parts.index('lora')
        data_name = '_'.join(parts[idx+1:])
    else:
        data_name = name
    
    if lora_name_filter == "all" or data_name == lora_name_filter:
        print(f"{data_name}|{path}")
EOF
}

# 测评单个LoRA的函数
evaluate_single_lora() {
    local DATA_NAME=$1
    local LORA_PATH=$2
    
    echo ""
    echo "========================================"
    echo "Evaluating: $DATA_NAME"
    echo "LoRA Path: $LORA_PATH"
    echo "========================================"
    
    # 检查LoRA路径是否存在
    if [ ! -d "$LORA_PATH" ]; then
        echo "[ERROR] LoRA path does not exist: $LORA_PATH"
        echo "$MODEL,$LORA_TYPE,$DATA_NAME,ERROR,ERROR" >> "$RESULTS_CSV"
        return
    fi
    
    # 设置测试数据路径
    TEST_FILE="$TEST_DATA_DIR/${DATA_NAME}_train.jsonl"
    if [ ! -f "$TEST_FILE" ]; then
        echo "[ERROR] Test data not found: $TEST_FILE"
        echo "$MODEL,$LORA_TYPE,$DATA_NAME,ERROR,ERROR" >> "$RESULTS_CSV"
        return
    fi
    
    # 如果指定了样本数，则截取测试数据
    if [ "$SAMPLES" != "-1" ]; then
        SAMPLE_FILE="$OUTPUT_BASE/${DATA_NAME}_sample.jsonl"
        head -n $SAMPLES "$TEST_FILE" > "$SAMPLE_FILE"
        TEST_FILE="$SAMPLE_FILE"
        echo "Using $SAMPLES samples from test data"
    fi
    
    # 输出目录
    RESULT_DIR="$OUTPUT_BASE/${DATA_NAME}_result"
    rm -rf "$RESULT_DIR"
    
    echo "Test Data: $TEST_FILE"
    echo "Result Dir: $RESULT_DIR"
    
    # 运行推理
    swift infer \
        --ckpt_dir "$LORA_PATH" \
        --val_dataset "$TEST_FILE" \
        --model_type $MODEL_TYPE \
        --model_id_or_path $MODEL_PATH \
        --sft_type lora \
        --result_dir "$RESULT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Inference failed for $DATA_NAME"
        echo "$MODEL,$LORA_TYPE,$DATA_NAME,ERROR,ERROR" >> "$RESULTS_CSV"
        return
    fi
    
    # 找到结果文件
    RESULT_JSONL=$(find "$RESULT_DIR" -name "*.jsonl" 2>/dev/null | head -n 1)
    if [ -z "$RESULT_JSONL" ]; then
        echo "[ERROR] No result file found for $DATA_NAME"
        echo "$MODEL,$LORA_TYPE,$DATA_NAME,ERROR,ERROR" >> "$RESULTS_CSV"
        return
    fi
    
    echo "Result file: $RESULT_JSONL"
    
    # 计算准确率
    EVAL_OUTPUT=$(python "$PROJECT_ROOT/evaluation/test_swift.py" --data_path "$RESULT_JSONL" 2>&1)
    
    # 提取准确率
    STEP_ACC=$(echo "$EVAL_OUTPUT" | grep "Step-level accuracy" | grep -oP '\d+\.\d+')
    EPISODE_ACC=$(echo "$EVAL_OUTPUT" | grep "Episode-level accuracy" | grep -oP '\d+\.\d+')
    
    if [ -z "$STEP_ACC" ]; then
        STEP_ACC="N/A"
    fi
    if [ -z "$EPISODE_ACC" ]; then
        EPISODE_ACC="N/A"
    fi
    
    echo ""
    echo ">>> Results for $DATA_NAME:"
    echo "    Step-level accuracy: ${STEP_ACC}%"
    echo "    Episode-level accuracy: ${EPISODE_ACC}%"
    
    # 保存到CSV
    echo "$MODEL,$LORA_TYPE,$DATA_NAME,$STEP_ACC,$EPISODE_ACC" >> "$RESULTS_CSV"
}

# 主循环：获取并测评所有LoRA
echo ""
echo "Fetching LoRA configurations..."

LORA_INFO=$(get_lora_info "$LORA_NAME")

if [ -z "$LORA_INFO" ]; then
    echo "Error: No LoRA found with name '$LORA_NAME'"
    exit 1
fi

# 统计总数
TOTAL_LORAS=$(echo "$LORA_INFO" | wc -l)
CURRENT=0

echo "Found $TOTAL_LORAS LoRA(s) to evaluate"
echo ""

# 遍历每个LoRA进行测评
while IFS='|' read -r DATA_NAME LORA_PATH; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "========================================================================"
    echo "[$CURRENT/$TOTAL_LORAS] Processing: $DATA_NAME"
    echo "========================================================================"
    
    evaluate_single_lora "$DATA_NAME" "$LORA_PATH"
    
done <<< "$LORA_INFO"

# 输出汇总
echo ""
echo "========================================================================"
echo "Evaluation Complete!"
echo "========================================================================"
echo ""
echo "Results Summary:"
cat "$RESULTS_CSV" | column -t -s','
echo ""
echo "Results saved to: $RESULTS_CSV"
echo "Full output saved to: $OUTPUT_BASE"
echo "========================================================================"
