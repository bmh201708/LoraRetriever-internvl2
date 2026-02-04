# LoraRetriever-InternVL2

Implementation of LoraRetriever paper using InternVL2 as base model with Swift framework.

## Features

- **Embedding-based LoRA Retrieval**: Uses jina-embeddings-v4 for semantic matching
- **Mixture/Fusion Composition**: Output-level (Mixture) and parameter-level (Fusion) LoRA combination
- **InternVL2 Integration**: Full support for InternVL2-2B multimodal model

## Installation

```bash
conda activate Lretriever
pip install -e .
```

## Usage

```bash
# Run inference
python infer_lora_retriever.py --test_data data/Val_100.jsonl --top_k 3 --merge_method mixture

# Test retriever
python tests/test_retriever.py
```

## Structure

- `lora_retriever/`: Core retrieval and composition module
- `swift/`: Modified Swift framework with LoRA mixture support
- `config/`: LoRA configuration files
- `data/`: Test data and embeddings
