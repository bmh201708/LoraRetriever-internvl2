#!/bin/bash
# Generate embeddings for all app/category LoRA datasets from FedMABench data-new.
# Uses first N training samples per dataset, with BOTH text and image embeddings.
# Usage: bash scripts/generate_missing_embeddings.sh

set -euo pipefail

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EMBED_SCRIPT="/home/hmpiao/hmpiao/jinyike/lorahub-for-fedmabench/scripts/embed.py"
JINA_MODEL="/home/hmpiao/hmpiao/jina-embeddings-v4"
FEDMA_DATA_ROOT="/home/hmpiao/hmpiao/jinyike/FedMABench/data-new"
APP_DATA_DIR="${FEDMA_DATA_ROOT}/data_by_app"
CATEGORY_DATA_DIR="${FEDMA_DATA_ROOT}/data_by_category"
OUT_APP_BASE="${PROJECT_ROOT}/data/embeddings/lora_app"
OUT_CATEGORY_BASE="${PROJECT_ROOT}/data/embeddings/lora_category"
TEMP_DIR="${PROJECT_ROOT}/data/temp_jsonl"

# Embedding parameters
NUM_SAMPLES="${NUM_SAMPLES:-20}"
JINA_MAX_PIXELS="${JINA_MAX_PIXELS:-100000}"
JINA_IMAGE_BATCH_SIZE="${JINA_IMAGE_BATCH_SIZE:-1}"

echo "=========================================="
echo "Generating LoRA Embeddings (all app + category)"
echo "=========================================="
echo "Jina Model: ${JINA_MODEL}"
echo "Data Root: ${FEDMA_DATA_ROOT}"
echo "Samples per dataset: ${NUM_SAMPLES}"
echo "Image max pixels: ${JINA_MAX_PIXELS}"
echo "Image batch size: ${JINA_IMAGE_BATCH_SIZE}"
echo "=========================================="

mkdir -p "${OUT_APP_BASE}" "${OUT_CATEGORY_BASE}" "${TEMP_DIR}"

# Activate conda environment
source /home/hmpiao/miniconda3/etc/profile.d/conda.sh
conda activate /data1/hmpiao/tmp/envs/Lretriever

run_embed_with_patch() {
    local input_jsonl="$1"
    local out_dir="$2"
    local out_prefix="$3"

    # Patch torch.library.wrap_triton before running embed.py
    python - "${EMBED_SCRIPT}" \
        --input_jsonl "${input_jsonl}" \
        --model_path "${JINA_MODEL}" \
        --out_dir "${out_dir}" \
        --out_prefix "${out_prefix}" \
        --text_task retrieval \
        --text_prompt_name query \
        --image_task retrieval \
        --text_source user \
        --image_batch_size "${JINA_IMAGE_BATCH_SIZE}" \
        --max_pixels "${JINA_MAX_PIXELS}" \
        --dtype float16 \
        --device cuda <<'PY'
import runpy
import sys
import torch

embed_script = sys.argv[1]
embed_args = sys.argv[2:]

if not hasattr(torch.library, "wrap_triton"):
    class _TritonKernelWrapper:
        def __init__(self, kernel):
            self.kernel = kernel
        def __getitem__(self, grid):
            return self.kernel[grid]
        def __call__(self, *args, **kwargs):
            return self.kernel(*args, **kwargs)

    def _dummy_wrap_triton(kernel):
        return _TritonKernelWrapper(kernel)

    torch.library.wrap_triton = _dummy_wrap_triton

sys.argv = [embed_script] + embed_args
runpy.run_path(embed_script, run_name="__main__")
PY
}

prepare_subset_jsonl() {
    local input_jsonl="$1"
    local output_jsonl="$2"
    local max_samples="$3"

    python - "$input_jsonl" "$output_jsonl" "$max_samples" <<'PY'
import json
import os
import re
import sys

inp, outp, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
input_dir = os.path.dirname(inp)

count = 0
with_text = 0
with_images = 0

def strip_image_tags(text: str) -> str:
    lines = str(text).splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        if s == "<image>" or s == "":
            continue
        kept.append(ln)
    return "\n".join(kept).strip()

with open(inp, "r", encoding="utf-8") as fin, open(outp, "w", encoding="utf-8") as fout:
    for line in fin:
        if count >= n:
            break
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        # Convert messages -> conversations for embed.py compatibility
        if "conversations" not in obj and isinstance(obj.get("messages"), list):
            conv = []
            for m in obj["messages"]:
                role = str(m.get("role", "")).strip().lower()
                content = m.get("content", "")
                if role == "user":
                    conv.append({"from": "user", "value": content})
                elif role == "assistant":
                    conv.append({"from": "assistant", "value": content})
            obj["conversations"] = conv

        # Normalize image paths to absolute local paths when possible
        imgs = obj.get("images") or []
        fixed_imgs = []
        for p in imgs:
            if not isinstance(p, str) or not p:
                continue
            if p.startswith("http://") or p.startswith("https://"):
                fixed_imgs.append(p)
            elif os.path.isabs(p):
                fixed_imgs.append(p)
            else:
                fixed_imgs.append(os.path.normpath(os.path.join(input_dir, p)))
        obj["images"] = fixed_imgs

        # Basic stats
        user_text = ""
        for c in obj.get("conversations", []):
            if str(c.get("from", "")).lower().strip() == "user":
                user_text = strip_image_tags(c.get("value", ""))
                break
        if user_text:
            with_text += 1
        if fixed_imgs:
            with_images += 1

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        count += 1

print(f"prepared_samples={count}, with_text={with_text}, with_images={with_images}")
PY
}

run_group() {
    local group_name="$1"
    local input_dir="$2"
    local output_base="$3"

    mapfile -t files < <(find "$input_dir" -maxdepth 1 -type f -name '*_train.jsonl' | sort)

    echo ""
    echo ">>> Processing group: ${group_name} (${#files[@]} datasets)"
    echo "----------------------------------------"

    for input_jsonl in "${files[@]}"; do
        local name
        name="$(basename "${input_jsonl}" _train.jsonl)"
        local out_dir="${output_base}/${name}"
        local out_prefix="${name}_jina_v4_emb"
        local temp_jsonl="${TEMP_DIR}/${group_name}_${name}_train_first${NUM_SAMPLES}.jsonl"

        mkdir -p "${out_dir}"

        echo ""
        echo "[${group_name}] ${name}"
        echo "input: ${input_jsonl}"
        echo "out:   ${out_dir}/${out_prefix}.mean.npy"

        prepare_subset_jsonl "${input_jsonl}" "${temp_jsonl}" "${NUM_SAMPLES}"

        run_embed_with_patch "${temp_jsonl}" "${out_dir}" "${out_prefix}"

        if [ -f "${out_dir}/${out_prefix}.mean.npy" ]; then
            echo "[SUCCESS] ${group_name}/${name}"
        else
            echo "[ERROR] Failed: ${group_name}/${name}"
        fi
    done
}

run_group "app" "${APP_DATA_DIR}" "${OUT_APP_BASE}"
run_group "category" "${CATEGORY_DATA_DIR}" "${OUT_CATEGORY_BASE}"

rm -f "${TEMP_DIR}"/app_*_train_first"${NUM_SAMPLES}".jsonl "${TEMP_DIR}"/category_*_train_first"${NUM_SAMPLES}".jsonl 2>/dev/null || true
rmdir "${TEMP_DIR}" 2>/dev/null || true

echo ""
echo "=========================================="
echo "All app/category embeddings generated!"
echo "App output: ${OUT_APP_BASE}"
echo "Category output: ${OUT_CATEGORY_BASE}"
echo "=========================================="
