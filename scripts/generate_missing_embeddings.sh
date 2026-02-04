#!/bin/bash
# Generate missing embeddings for LoRA (limited to first 20 samples)
# Usage: bash scripts/generate_missing_embeddings.sh

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EMBED_SCRIPT="/home/hmpiao/hmpiao/jinyike/lorahub-for-fedmabench/scripts/embed.py"
JINA_MODEL="/home/hmpiao/hmpiao/jina-embeddings-v4"
FEDMA_DATA="/home/hmpiao/hmpiao/jinyike/FedMABench/data_by_app"
OUTPUT_BASE="${PROJECT_ROOT}/data/embeddings/lora_app"
TEMP_DIR="${PROJECT_ROOT}/data/temp_jsonl"

# Number of samples to use for embedding
NUM_SAMPLES=20

# Missing apps (those without embeddings)
MISSING_APPS=("adidas" "decathlon" "etsy" "calendar" "google_maps" "kitchen_stories")

echo "=========================================="
echo "Generating Missing LoRA Embeddings"
echo "=========================================="
echo "Jina Model: ${JINA_MODEL}"
echo "Output Dir: ${OUTPUT_BASE}"
echo "Samples per LoRA: ${NUM_SAMPLES}"
echo "Missing Apps: ${MISSING_APPS[@]}"
echo "=========================================="

# Create output and temp directories
for app in "${MISSING_APPS[@]}"; do
    mkdir -p "${OUTPUT_BASE}/${app}"
done
mkdir -p "${TEMP_DIR}"

# Activate conda environment
source ~/miniconda3/bin/activate Lretriever

# Generate embeddings for each missing app
for app in "${MISSING_APPS[@]}"; do
    echo ""
    echo ">>> Processing: ${app}"
    echo "----------------------------------------"
    
    INPUT_JSONL="${FEDMA_DATA}/${app}_train.jsonl"
    TEMP_JSONL="${TEMP_DIR}/${app}_train_first${NUM_SAMPLES}.jsonl"
    OUT_DIR="${OUTPUT_BASE}/${app}"
    OUT_PREFIX="${app}_jina_v4_emb"
    
    if [ ! -f "$INPUT_JSONL" ]; then
        echo "[WARNING] Input file not found: ${INPUT_JSONL}"
        continue
    fi
    
    # Extract first N samples
    echo "Extracting first ${NUM_SAMPLES} samples..."
    head -n ${NUM_SAMPLES} "${INPUT_JSONL}" > "${TEMP_JSONL}"
    echo "Created temp file with $(wc -l < "${TEMP_JSONL}") samples"
    
    # Generate embedding using the limited dataset
    python "${EMBED_SCRIPT}" \
        --input_jsonl "${TEMP_JSONL}" \
        --model_path "${JINA_MODEL}" \
        --out_dir "${OUT_DIR}" \
        --out_prefix "${OUT_PREFIX}" \
        --text_task retrieval \
        --text_prompt_name query \
        --dtype float16 \
        --device cuda
    
    # Check if output was created
    if [ -f "${OUT_DIR}/${OUT_PREFIX}.mean.npy" ]; then
        echo "[SUCCESS] Generated embedding for ${app}"
    else
        echo "[ERROR] Failed to generate embedding for ${app}"
    fi
    
    # Cleanup temp file
    rm -f "${TEMP_JSONL}"
done

# Cleanup temp directory
rmdir "${TEMP_DIR}" 2>/dev/null || true

echo ""
echo "=========================================="
echo "Embedding generation complete!"
echo "=========================================="
