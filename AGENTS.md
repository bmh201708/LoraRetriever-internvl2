# Repository Guidelines

## Project Structure & Module Organization
- `infer_lora_retriever.py`: main retrieve-then-compose entrypoint (supports `step` and `episode` dialog modes).
- `lora_retriever/`:
  - `retriever.py`: Jina embedding-based LoRA retrieval.
  - `composition.py`: composition logic for `mixture` and `fusion`.
- `config/`: LoRA metadata (`app_loras_config_*.json`, `category_loras_config_*.json`) with `lora_path` and `embedding_path`.
- `evaluation/`: evaluation runner scripts and metric script:
  - `run_eval_lora_retriever.sh` (single dataset, runs step+episode),
  - `run_eval_id.sh`, `run_eval_ood.sh` (batch),
  - `evaluate_inference.py` (compute step/episode accuracy from result JSONL).
- `scripts/`: helper scripts such as `generate_missing_embeddings.sh`.
- `tests/`: functional smoke scripts (`test_retriever.py`, `test_all_loras.py`, `test_e2e_inference.py`).
- `data/`, `output/`, `evaluation/results/`: local inputs, embeddings, inference outputs, and evaluation artifacts.

## Build, Test, and Development Commands
- Activate env:
  - `source /home/hmpiao/miniconda3/etc/profile.d/conda.sh`
  - `conda activate /data1/hmpiao/tmp/envs/Lretriever`
- Install editable package: `pip install -e .`
- Run inference (single JSONL):
  `python infer_lora_retriever.py --model_type qwen2-vl-2b-instruct --test_data data/test_data_by_app/amazon_train.jsonl --dialog_mode step --merge_method mixture --top_k 3 --gpu_id 0`
- Run step+episode eval for one app:
  `bash evaluation/run_eval_lora_retriever.sh --test_input amazon --model qwen2b --gpu_id 0`
- Batch eval:
  `bash evaluation/run_eval_id.sh --model qwen2b --gpu_id 0`
- Run retrieval test: `python tests/test_retriever.py`
- Run end-to-end smoke test: `python tests/test_e2e_inference.py --num_samples 5`

## Coding Style & Naming Conventions
- Python style is configured in `setup.cfg`.
- Line length: `120` (`flake8`, `yapf`, `isort` aligned).
- Use 4-space indentation and clear snake_case names (`load_lora_configs`, `run_inference`).
- Keep retrieval/composition in `lora_retriever/`; keep CLI orchestration in entry scripts.

## Testing Guidelines
- Tests are script-style Python files under `tests/` (not strict pytest fixtures).
- Naming pattern: `test_*.py`.
- For retrieval/composition changes, validate both unit behavior (`tests/test_retriever.py`) and E2E (`tests/test_e2e_inference.py`).
- For evaluation changes, run `evaluation/run_eval_lora_retriever.sh` and archive `summary_paths.txt`.

## Commit & Pull Request Guidelines
- Existing history uses short, task-focused messages (often Chinese), e.g. `修复fusion`, `评测脚本`.
- Follow the same style: one commit per logical change, subject line in imperative style.
- PRs should include:
  - purpose and scope
  - key commands run and results
  - changed config/model paths (if any)
  - sample output path (e.g., `output/...jsonl`) for evaluation-impacting changes.

## Security & Configuration Tips
- This repo uses many absolute local paths (model/data/cache). Prefer passing overrides via CLI args or env vars.
- `infer_lora_retriever.py` can restrict app candidate pool via `APP=(...)`; keep names consistent with `lora_name` in config.
- Do not commit secrets, private tokens, or large generated artifacts under `output/` or `evaluation/results/`.
