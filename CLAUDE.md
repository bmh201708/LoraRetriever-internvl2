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

# Debug mode with similarity output
python infer_lora_retriever.py --test_data data/Val_100.jsonl --debug --show_similarities --num_samples 5

# Run unit tests
python tests/test_retriever.py
python tests/test_all_loras.py
python tests/test_e2e_inference.py --num_samples 5

# Run evaluation via shell script
./scripts/run_eval.sh --test_data data/Val_100.jsonl --top_k 3 --merge_method mixture
```

## Environment Variables

Set before running inference:
```bash
export MAX_PIXELS=150000          # Image processing resolution
export MAX_NUM=12                 # Maximum image count per sample
export CUDA_VISIBLE_DEVICES=0     # GPU selection
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
```

## Architecture

### Core Components

**lora_retriever/** - Core retrieval and composition module
- `retriever.py`: Embedding-based LoRA retrieval using jina-embeddings-v4 and cosine similarity
- `composition.py`: MixtureComposer (output-level weighted average) and FusionComposer (parameter-level merging)

**swift/** - Modified Alibaba Swift framework with mixture mode support
- `llm/utils/`: Model loading (`get_model_tokenizer`), templates (`get_template`), inference utilities
- `tuners/lora_layers.py`: Modified LoRA layers with `LOGOMixtureContext` for per-sample adapter weighting
- `tuners/`: LoRA implementation and adapter management

**config/** - LoRA configuration files (JSON)
- `app_loras_config_internvl2.json`: 14 app-specific LoRAs (Adidas, Amazon, Gmail, etc.)
- `category_loras_config_internvl2.json`: 5 category LoRAs (Shopping, Entertainment, etc.)

**data/** - Test data and pre-computed embeddings
- `embeddings/lora_app/`: Pre-computed app LoRA embeddings (.npy files)
- `embeddings/lora_category/`: Pre-computed category LoRA embeddings
- `Val_100.jsonl`: 100-sample validation dataset

### Key Import Paths

```python
from lora_retriever import LoraRetriever, LoraRetrieverConfig, MixtureComposer, FusionComposer, CompositionStrategy
from swift.tuners import Swift
from swift.llm.utils import get_model_tokenizer, get_template, inference
from swift.utils import get_logger, append_to_jsonl, seed_everything
```

### Composition Strategies

- **Mixture**: All adapters loaded in memory, different weights applied per sample at output level (`x' = Σ_j B_j * A_j * x`). More flexible, higher memory usage.
- **Fusion**: Parameters merged into single adapter before inference. Lower memory, faster inference, less flexible.

## Data Format

Input JSONL files use this format:
```json
{
  "images": ["path/to/img1.png", "path/to/img2.png"],
  "query": "<image>\n<image>\nText description of task",
  "response": "Expected action sequence",
  "episode_id": "000058"
}
```

## Configuration Format

LoRA configs in JSON (`config/*.json`):
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
- `MODEL_PATHS`: Base model paths (InternVL2-2B, Qwen2-VL-7B-Instruct)
- LoRA checkpoint paths in config files

## Swift Framework Modifications

Key modifications to the Swift framework for mixture mode support:
1. `swift/tuners/lora_layers.py`: Added `LOGOMixtureContext` for per-sample adapter weighting
2. Multi-adapter loading via `Swift.from_pretrained()` with `adapter_name` parameter
3. `lora_mapping` tensor control in inference for batch-level LoRA selection
