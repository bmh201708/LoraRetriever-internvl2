#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/run_eval_lora_retriever.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "[ERROR] target script not found: $TARGET"
  exit 1
fi

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gou_id)
      # Compatibility for typo: gou_id -> gpu_id
      ARGS+=(--gpu_id "$2")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

bash "$TARGET" "${ARGS[@]}"
