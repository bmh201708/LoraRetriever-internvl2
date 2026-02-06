#!/bin/bash
#
# LoraRetriever 推理与评测脚本
# 先运行 infer_lora_retriever.py 完成推理，再使用 test_swift.py 评测结果
#

set -e  # 遇到错误时退出

# ==============================================================================
# 环境配置
# ==============================================================================

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Lretriever

# 设置环境变量 (CUDA_VISIBLE_DEVICES 会在参数解析后设置)
export MAX_PIXELS=${MAX_PIXELS:-100000}
export MAX_NUM=${MAX_NUM:-12}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=""  # 禁用 Triton 编译，避免兼容性问题

# Jina Embeddings 显存优化参数
export JINA_MAX_PIXELS=${JINA_MAX_PIXELS:-100000}  # 图像分辨率限制（默认约316x316）
export JINA_IMAGE_BATCH_SIZE=${JINA_IMAGE_BATCH_SIZE:-1}  # 图像编码批量大小

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ==============================================================================
# 默认参数
# ==============================================================================

# 推理相关参数
MODEL_TYPE=${MODEL_TYPE:-internvl2-2b}
TEST_DATA=${TEST_DATA:-data/Val_100.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-output/lora_retriever_results}
TOP_K=${TOP_K:-3}
MERGE_METHOD=${MERGE_METHOD:-mixture}
LORA_TYPE=${LORA_TYPE:-all}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
TEMPERATURE=${TEMPERATURE:-0.0}

# 调试参数
DEBUG=${DEBUG:-false}
NUM_SAMPLES=${NUM_SAMPLES:-}
SHOW_SIMILARITIES=${SHOW_SIMILARITIES:-false}

# GPU 设置 (默认为空，使用环境变量或系统默认)
GPU_ID=${GPU_ID:-}

# 评测参数
SKIP_INFERENCE=${SKIP_INFERENCE:-false}
INFERENCE_RESULT=${INFERENCE_RESULT:-}

# 批量评测参数
BATCH_MODE=${BATCH_MODE:-false}
APP_ONLY=${APP_ONLY:-false}
CATEGORY_ONLY=${CATEGORY_ONLY:-false}
DRY_RUN=${DRY_RUN:-false}

# ==============================================================================
# 参数解析
# ==============================================================================

usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "推理参数:"
    echo "  --model_type TYPE       模型类型 (默认: internvl2-2b)"
    echo "  --test_data PATH        测试数据路径 (默认: data/Val_100.jsonl)"
    echo "  --output_dir DIR        输出目录 (默认: output/lora_retriever_results)"
    echo "  --top_k K               选择 top-k 个 LoRA (默认: 3)"
    echo "  --merge_method METHOD   合并方法: mixture 或 fusion (默认: mixture)"
    echo "  --lora_type TYPE        LoRA 类型: all, app, category (默认: all)"
    echo "  --max_new_tokens N      最大生成 token 数 (默认: 512)"
    echo "  --temperature T         温度参数 (默认: 0.0)"
    echo ""
    echo "调试参数:"
    echo "  --debug                 开启调试模式"
    echo "  --num_samples N         只处理 N 个样本"
    echo "  --show_similarities     显示所有 LoRA 的相似度分数"
    echo "  --gpu_id ID             使用指定的 GPU (设置 CUDA_VISIBLE_DEVICES)"
    echo ""
    echo "评测参数:"
    echo "  --skip_inference        跳过推理，直接评测"
    echo "  --inference_result PATH 指定推理结果文件（与 --skip_inference 配合使用）"
    echo ""
    echo "批量评测参数:"
    echo "  --batch                 批量评测模式（评测所有 app 和 category 数据集）"
    echo "  --app_only              只评测 app 级别（与 --batch 配合使用）"
    echo "  --category_only         只评测 category 级别（与 --batch 配合使用）"
    echo "  --dry_run               只打印命令，不实际执行（与 --batch 配合使用）"
    echo ""
    echo "环境变量 (可选):"
    echo "  MAX_NUM                 推理时最大图片数 (默认: 12，用于避免 OOM)"
    echo "  MAX_PIXELS              单张图片最大像素数 (默认: 100000)"
    echo "  JINA_MAX_PIXELS         Jina 编码最大像素数 (默认: 100000，约 316x316)"
    echo "  JINA_IMAGE_BATCH_SIZE   Jina 编码批量大小 (默认: 1，避免 OOM)"
    echo ""
    echo "示例:"
    echo "  $0 --test_data data/Val_100.jsonl --top_k 3 --merge_method mixture"
    echo "  $0 --skip_inference --inference_result output/results.jsonl"
    echo "  MAX_NUM=8 $0 --test_data data/Val_100.jsonl --top_k 3  # 限制最多8张图"
    echo ""
    echo "  # 批量评测所有 app 和 category 数据集"
    echo "  $0 --batch --gpu_id 5 --top_k 3 --merge_method mixture"
    echo "  $0 --batch --app_only --gpu_id 5"
    echo "  $0 --batch --category_only --gpu_id 5"
    echo "  $0 --batch --dry_run"
    echo "  $0 --batch --skip_inference --output_dir output/retriever_eval_xxx"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        --merge_method)
            MERGE_METHOD="$2"
            shift 2
            ;;
        --lora_type)
            LORA_TYPE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --show_similarities)
            SHOW_SIMILARITIES=true
            shift
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --skip_inference)
            SKIP_INFERENCE=true
            shift
            ;;
        --inference_result)
            INFERENCE_RESULT="$2"
            shift 2
            ;;
        --batch)
            BATCH_MODE=true
            shift
            ;;
        --app_only)
            APP_ONLY=true
            shift
            ;;
        --category_only)
            CATEGORY_ONLY=true
            shift
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项: $1"
            usage
            ;;
    esac
done

# ==============================================================================
# 设置 CUDA_VISIBLE_DEVICES (在参数解析后)
# ==============================================================================

if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
elif [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# ==============================================================================
# 批量评测模式 (--batch)
# ==============================================================================

if [ "$BATCH_MODE" = true ]; then
    echo ""
    echo "[批量评测模式] 调用 evaluation/evaluate_all.py ..."
    echo "=============================================="

    EVAL_CMD="python evaluation/evaluate_all.py \
        --model_type $MODEL_TYPE \
        --gpu_id ${GPU_ID:-$CUDA_VISIBLE_DEVICES} \
        --top_k $TOP_K \
        --merge_method $MERGE_METHOD \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE"

    if [ -n "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "output/lora_retriever_results" ]; then
        EVAL_CMD="$EVAL_CMD --output_dir $OUTPUT_DIR"
    fi
    if [ "$APP_ONLY" = true ]; then
        EVAL_CMD="$EVAL_CMD --app_only"
    fi
    if [ "$CATEGORY_ONLY" = true ]; then
        EVAL_CMD="$EVAL_CMD --category_only"
    fi
    if [ "$SKIP_INFERENCE" = true ]; then
        EVAL_CMD="$EVAL_CMD --skip_inference"
    fi
    if [ "$DEBUG" = true ]; then
        EVAL_CMD="$EVAL_CMD --debug"
    fi
    if [ -n "$NUM_SAMPLES" ]; then
        EVAL_CMD="$EVAL_CMD --num_samples $NUM_SAMPLES"
    fi
    if [ "$SHOW_SIMILARITIES" = true ]; then
        EVAL_CMD="$EVAL_CMD --show_similarities"
    fi
    if [ "$DRY_RUN" = true ]; then
        EVAL_CMD="$EVAL_CMD --dry_run"
    fi

    echo "执行命令: $EVAL_CMD"
    echo ""
    eval $EVAL_CMD
    exit $?
fi

# ==============================================================================
# 打印配置信息 (单文件模式)
# ==============================================================================

echo "=============================================="
echo "LoraRetriever 推理与评测"
echo "=============================================="
echo "项目根目录: $PROJECT_ROOT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""
echo "推理参数:"
echo "  模型类型: $MODEL_TYPE"
echo "  测试数据: $TEST_DATA"
echo "  输出目录: $OUTPUT_DIR"
echo "  Top-K: $TOP_K"
echo "  合并方法: $MERGE_METHOD"
echo "  LoRA类型: $LORA_TYPE"
echo "  最大生成Token: $MAX_NEW_TOKENS"
echo "  温度: $TEMPERATURE"
echo ""
if [ "$DEBUG" = true ]; then
    echo "调试模式: 开启"
fi
if [ -n "$NUM_SAMPLES" ]; then
    echo "样本数限制: $NUM_SAMPLES"
fi
if [ -n "$GPU_ID" ]; then
    echo "指定GPU: $GPU_ID"
fi
echo "=============================================="

# ==============================================================================
# 步骤1: 运行推理
# ==============================================================================

if [ "$SKIP_INFERENCE" = true ]; then
    echo ""
    echo "[跳过推理] 使用已有结果进行评测"
    
    if [ -z "$INFERENCE_RESULT" ]; then
        echo "错误: 使用 --skip_inference 时必须指定 --inference_result"
        exit 1
    fi
    
    if [ ! -f "$INFERENCE_RESULT" ]; then
        echo "错误: 推理结果文件不存在: $INFERENCE_RESULT"
        exit 1
    fi
    
    RESULT_FILE="$INFERENCE_RESULT"
else
    echo ""
    echo "[步骤1/2] 开始运行推理..."
    echo "=============================================="
    
    # 构建推理命令
    INFER_CMD="python infer_lora_retriever.py \
        --model_type $MODEL_TYPE \
        --test_data $TEST_DATA \
        --output_dir $OUTPUT_DIR \
        --top_k $TOP_K \
        --merge_method $MERGE_METHOD \
        --lora_type $LORA_TYPE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --temperature $TEMPERATURE"
    
    if [ "$DEBUG" = true ]; then
        INFER_CMD="$INFER_CMD --debug"
    fi
    
    if [ -n "$NUM_SAMPLES" ]; then
        INFER_CMD="$INFER_CMD --num_samples $NUM_SAMPLES"
    fi
    
    if [ "$SHOW_SIMILARITIES" = true ]; then
        INFER_CMD="$INFER_CMD --show_similarities"
    fi
    
    if [ -n "$GPU_ID" ]; then
        INFER_CMD="$INFER_CMD --gpu_id $GPU_ID"
    fi
    
    echo "执行命令: $INFER_CMD"
    echo ""
    
    # 运行推理
    eval $INFER_CMD
    
    # 获取最新生成的结果文件
    MODEL_SUFFIX="internvl2"
    if [[ "$MODEL_TYPE" == *"qwen2"* ]]; then
        MODEL_SUFFIX="qwen2vl"
    fi
    
    RESULT_FILE=$(ls -t "$OUTPUT_DIR"/retriever_results_${MODEL_SUFFIX}_${MERGE_METHOD}_k${TOP_K}_*.jsonl 2>/dev/null | head -n 1)
    
    if [ -z "$RESULT_FILE" ]; then
        echo "错误: 未找到推理结果文件"
        exit 1
    fi
    
    echo ""
    echo "推理完成！结果保存在: $RESULT_FILE"
fi

# ==============================================================================
# 步骤2: 运行评测
# ==============================================================================

echo ""
echo "[步骤2/2] 开始运行评测..."
echo "=============================================="
echo "评测文件: $RESULT_FILE"
echo ""

# 运行评测
python evaluation/test_swift.py --data_path "$RESULT_FILE"

# ==============================================================================
# 完成
# ==============================================================================

echo ""
echo "=============================================="
echo "推理与评测完成！"
echo "=============================================="
echo "推理结果: $RESULT_FILE"
echo "评测日志: ${RESULT_FILE%.jsonl}.log"
echo "=============================================="
