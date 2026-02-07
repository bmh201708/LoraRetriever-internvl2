#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
export MAX_PIXELS=100000
export MAX_NUM=12

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Lretriever

# Run inference with Qwen2-VL on a small sample
python infer_lora_retriever.py \
    --model_type qwen2-vl-7b-instruct \
    --test_data data/Val_100.jsonl \
    --top_k 3 \
    --merge_method mixture \
    --debug \
    --num_samples 2 \
    --show_similarities

