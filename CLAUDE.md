# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LoraRetriever-InternVL2 implements the LoraRetriever paper for dynamic, input-aware LoRA retrieval and composition with multimodal vision-language models. Instead of using a single fine-tuned LoRA, it maintains multiple specialized LoRA adapters and dynamically selects the best-matching adapters based on query similarity.

**Core workflow**: Query + Images → Retriever (embedding similarity via jina-embeddings-v4) → Top-K LoRAs with Weights → Composer (mixture/fusion) → Base Model + Composed LoRAs → Response

## Key Commands

```bash
# Install (uses Lretriever conda environment)
conda activate Lretriever
pip install -e .

# Run inference with LoRA retrieval (mixture mode)
python infer_lora_retriever.py --test_data data/Val_100.jsonl --top_k 3 --merge_method mixture

# Run inference with fusion composition
python infer_lora_retriever.py --test_data data/Val_100.jsonl --top_k 5 --merge_method fusion

# Only app or category LoRAs
python infer_lora_retriever.py --test_data data/Val_100.jsonl --lora_type app
python infer_lora_retriever.py --test_data data/Val_100.jsonl --lora_type category

# Debug mode with similarity output
python infer_lora_retriever.py --test_data data/Val_100.jsonl --debug --show_similarities --num_samples 5

# GPU selection (default is gpu_id=5)
python infer_lora_retriever.py --gpu_id 0 --test_data data/Val_100.jsonl

# Run unit tests
python tests/test_retriever.py
python tests/test_all_loras.py
python tests/test_e2e_inference.py --num_samples 5

# Run evaluation via shell script
./scripts/run_eval.sh --test_data data/Val_100.jsonl --top_k 3 --merge_method mixture

# Run comprehensive evaluation across all 14 app + 5 category datasets
python evaluation/evaluate_all.py --merge_method mixture --top_k 3 --gpu_id 0

# Evaluate existing results only (skip inference)
./scripts/run_eval.sh --evaluate_only path/to/results.jsonl --threshold 0.6
```

## Environment Variables

Set before running inference (the script sets defaults if unset):
```bash
export MAX_PIXELS=100000          # Image processing resolution (default in code: 100000)
export MAX_NUM=12                 # Maximum image count per sample
export CUDA_VISIBLE_DEVICES=0     # GPU selection (also settable via --gpu_id flag)
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
```

## Code Style

Configured in `setup.cfg`: line length 120, yapf (pep8-based), isort, flake8. Ignored flake8 rules: F401, F403, F405, F821, W503, E251, W504, E126.

## Architecture

### Core Components

**lora_retriever/** - Core retrieval and composition module
- `retriever.py`: Embedding-based LoRA retrieval using jina-embeddings-v4 and cosine similarity. Uses `_call_with_supported_kwargs` to handle API version differences in the Jina model.
- `composition.py`: MixtureComposer (output-level weighted average) and FusionComposer (parameter-level merging).

**infer_lora_retriever.py** - Main inference entry point. Parses `--gpu_id` before torch import to set `CUDA_VISIBLE_DEVICES` early. Supports InternVL2-2B (default) and Qwen2-VL-7B-Instruct.

**swift/** - Modified Alibaba Swift framework (v2.6.0.dev0) with mixture mode support. This is a vendored fork; key modified files:
- `tuners/lora_layers.py`: Added `LOGOMixtureContext` global singleton for per-sample adapter weighting. Methods: `set()`, `is_mixture_mode()`, `reset()`.
- `tuners/base.py`: `SwiftModel` with `add_weighted_adapter()` for fusion and `set_active_adapters()` for adapter switching.
- `llm/utils/model.py`: Model loading (`get_model_tokenizer`) — very large file, avoid reading entirely.
- `llm/utils/template.py`: Chat templates (`get_template`) — very large file, avoid reading entirely.
- `llm/utils/utils.py`: Inference utilities including the `inference()` function.

**config/** - LoRA configuration files (JSON arrays). Each model has separate configs:
- `*_internvl2.json`: 14 app-specific + 5 category LoRAs for InternVL2-2B
- `*_qwen2vl.json`: Same LoRAs for Qwen2-VL-7B

**evaluation/** - Evaluation pipeline
- `evaluate_all.py`: Runs inference + evaluation across all 19 datasets (14 app + 5 category), generates `summary.md` report with step-level and episode-level accuracy.
- `eval_gpt.py`: GPT-based evaluation of generated responses.

### Key Import Paths

```python
from lora_retriever import LoraRetriever, LoraRetrieverConfig, MixtureComposer, FusionComposer, CompositionStrategy
from swift.tuners import Swift
from swift.llm.utils import get_model_tokenizer, get_template, inference
from swift.utils import get_logger, append_to_jsonl, seed_everything
```

### Composition Strategies

- **Mixture**: All adapters loaded in memory, different weights applied per sample at output level (`x' = Σ_j w_j * B_j * A_j * x`). More flexible, higher memory usage. Uses `lora_mapping` tensor (batch_size x num_adapters) passed to `inference()` with `merging_type='mixture'`.
- **Fusion**: Parameters merged into single adapter (`fused_lora`) before inference via `SwiftModel.add_weighted_adapter()`. Lower memory, faster inference, less flexible.

### Inference Flow (mixture mode)

1. `LoraRetriever.retrieve_with_weights()` → returns `(selected_lora_names, weights)`
2. `MixtureComposer.compose()` → creates `lora_mapping` tensor
3. `inference(model, template, query, ..., merging_type='mixture', lora_mapping=lora_mapping, mixture_adapter_names=adapter_names)` → generates response
4. Inside Swift, `LOGOMixtureContext` applies per-adapter weights at the LoRA layer level

### Inference Flow (fusion mode)

1. `LoraRetriever.retrieve_with_weights()` → returns `(selected_lora_names, weights)`
2. `FusionComposer.compose()` → creates merged `fused_lora` adapter via parameter averaging
3. `inference(model, template, query, ...)` → generates with fused adapter (standard inference, no special flags)
4. `FusionComposer.cleanup()` → removes fused adapter after all samples processed

### Known Issue: Fusion Mode Performance Degradation

In fusion mode, `set_active_adapters()` deactivates all non-active adapters. Each new sample creates a new `fused_N` adapter, so the number of adapters to deactivate grows linearly with processed samples — O((M+N)*L) where M=original LoRAs, N=samples processed, L=LoRA layers. This causes severe slowdown after the first sample. The fix is to reuse a single fused adapter and update its parameters in-place rather than creating new ones. See `docs/fusion_performance_issue.md` for full analysis.

## Data Format

Input JSONL files:
```json
{
  "images": ["path/to/img1.png", "path/to/img2.png"],
  "query": "<image>\n<image>\nText description of task",
  "response": "Expected action sequence",
  "episode_id": "000058"
}
```

Image paths may be relative (resolved against project root) or use `./../` prefix (resolved relative to parent directory).

LoRA configs in `config/*.json` are arrays of:
```json
{
  "lora_name": "app_lora_adidas",
  "lora_path": "/path/to/lora/checkpoint",
  "embedding_path": "data/embeddings/lora_app/adidas/adidas_jina_v4_emb.mean.npy",
  "description": "InternVL2-2B LoRA for Adidas app"
}
```

## Hardcoded Paths (in infer_lora_retriever.py)

The following paths are hardcoded and may need adjustment:
- `JINA_MODEL_PATH`: jina-embeddings-v4 model location (`/home/hmpiao/hmpiao/jina-embeddings-v4`)
- `MODEL_PATHS`: Base model paths (InternVL2-2B at `/home/hmpiao/hmpiao/InternVL2-2B-ModelScope/OpenGVLab/InternVL2-2B`, Qwen2-VL-7B at `/home/hmpiao/hmpiao/Qwen2-VL-7B-Instruct`)
- LoRA checkpoint paths in config JSON files
- Default `--gpu_id` is `5`
