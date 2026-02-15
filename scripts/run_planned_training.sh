#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"
CSV_DATE="${CSV_DATE:-$(date -u +%Y-%m-%d)}"
CSV_PATH="${CSV_PATH:-$ROOT_DIR/../project-brain/daily-briefs/EXPERIMENT_TEST_LOG_${CSV_DATE}.csv}"

if [[ ! -f "$CSV_PATH" ]]; then
  echo "CSV not found: $CSV_PATH" >&2
  echo "Set CSV_PATH or CSV_DATE to target a specific experiment log." >&2
  exit 1
fi

declare -a CMD=(
  "$PY_BIN"
  "$ROOT_DIR/scripts/run_training_from_csv.py"
  --csv
  "$CSV_PATH"
)

status_explicit=0
for arg in "$@"; do
  case "$arg" in
    --status|--status=*)
      status_explicit=1
      ;;
  esac
done

if [[ "$status_explicit" -eq 0 ]]; then
  CMD+=(--status planned)
fi

CMD+=("$@")
exec "${CMD[@]}"
