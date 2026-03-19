# Repository Guidelines

## Project Structure & Module Organization
- `infer_lora_retriever.py`: main entry for retrieve-then-compose inference.
- `lora_retriever/`: core logic.
  - `retriever.py`: embedding-based LoRA retrieval.
  - `composition.py`: `mixture` and `fusion` composition strategies.
- `config/`: LoRA metadata (`app_*`, `category_*`) with `lora_path` and `embedding_path`.
- `evaluation/`: evaluation pipeline (`test_swift.py`, `evaluate_all.py`).
- `scripts/`: runnable shell wrappers for inference/eval workflows.
- `tests/`: functional test scripts (`test_retriever.py`, `test_all_loras.py`, `test_e2e_inference.py`).
- `data/` and `output/`: local datasets, embeddings, and generated results.

## Build, Test, and Development Commands
- Activate env:
  - `source /home/hmpiao/miniconda3/etc/profile.d/conda.sh`
  - `conda activate /data1/hmpiao/tmp/envs/Lretriever`
- Install editable package: `pip install -e .`
- Run inference:  
  `python infer_lora_retriever.py --test_data data/Val_100.jsonl --top_k 3 --merge_method mixture`
- Run retrieval test: `python tests/test_retriever.py`
- Run end-to-end smoke test: `python tests/test_e2e_inference.py --num_samples 5`
- Run scripted pipeline: `bash scripts/run_inference_and_eval.sh --top_k 3 --merge_method mixture`

## Coding Style & Naming Conventions
- Python style is configured in `setup.cfg`.
- Line length: `120` (`flake8`, `yapf`, `isort` aligned).
- Use 4-space indentation and clear snake_case names (`load_lora_configs`, `run_inference`).
- Keep modules focused: retrieval logic in `lora_retriever/`, orchestration in scripts/entrypoints.

## Testing Guidelines
- Tests are script-style Python files under `tests/` (not strict pytest fixtures).
- Naming pattern: `test_*.py`.
- Add at least one runnable command example for new features.
- For retrieval/composition changes, validate both:
  - unit behavior (`tests/test_retriever.py`)
  - end-to-end path (`tests/test_e2e_inference.py`)

## Commit & Pull Request Guidelines
- Existing history uses short, task-focused messages (often Chinese), e.g. `修复fusion`, `评测脚本`.
- Follow the same style: one commit per logical change, subject line in imperative style.
- PRs should include:
  - purpose and scope
  - key commands run and results
  - changed config/model paths (if any)
  - sample output path (e.g., `output/...jsonl`) for evaluation-impacting changes.

## Security & Configuration Tips
- This repo contains hardcoded absolute model paths; keep machine-specific paths configurable when possible.
- Do not commit secrets, private tokens, or large generated artifacts under `output/`.
