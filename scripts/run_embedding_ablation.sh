#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -x "$PY_BIN" ]]; then
  echo "Python venv not found at $PY_BIN" >&2
  echo "Set PY_BIN or create the venv in $ROOT_DIR/.venv" >&2
  exit 1
fi

INPUT_DIR="${INPUT_DIR:-}"
if [[ -z "$INPUT_DIR" ]]; then
  for candidate in \
    "$ROOT_DIR/../.tmp/prodata-2025-plus-2026-01-clean" \
    "$ROOT_DIR/data/prodata" \
    "$ROOT_DIR/../.tmp/prodata-2025-clean" \
    "$ROOT_DIR/../.tmp/prodata-2025"
  do
    if [[ -d "$candidate" ]]; then
      INPUT_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "$INPUT_DIR" ]]; then
  echo "No input dir found. Set INPUT_DIR to a processed prodata directory." >&2
  exit 1
fi

CHAMPION_MAPPING_PATH="${CHAMPION_MAPPING_PATH:-}"
if [[ -z "$CHAMPION_MAPPING_PATH" ]]; then
  for candidate in \
    "$ROOT_DIR/data/ddragon/artifacts/champion-mapping/latest.json" \
    "$ROOT_DIR/../.tmp/champion-mapping/latest.json" \
    "$ROOT_DIR/../lol-ddragon-context-artifact-builder/data/ddragon/artifacts/champion-mapping/latest.json" \
    "$ROOT_DIR/../lol-ddragon-snapshot-cron/data/ddragon/artifacts/champion-mapping/latest.json"
  do
    if [[ -f "$candidate" ]]; then
      CHAMPION_MAPPING_PATH="$candidate"
      break
    fi
  done
fi

if [[ -z "$CHAMPION_MAPPING_PATH" ]]; then
  echo "No champion mapping found. Set CHAMPION_MAPPING_PATH to a mapping JSON." >&2
  exit 1
fi

EPOCHS="${EPOCHS:-20}"
SEED="${SEED:-42}"
DATASET_LABEL="${DATASET_LABEL:-Clean 2025 + Jan 2026}"
CATEGORY="${CATEGORY:-embedding-ablation}"
RUN_BASELINE="${RUN_BASELINE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/../.tmp/training-embedding-ablation-$(date -u +%Y%m%d_%H%M%S)}"

PUBLISH_DATA="${PUBLISH_DATA:-0}"
PUBLISH_DATA_DIR="${PUBLISH_DATA_DIR:-$ROOT_DIR/../draft-sage-training-data/public/training}"

mkdir -p "$OUTPUT_DIR"

run_variant() {
  local display_name="$1"
  shift
  echo "Running: $display_name"
  "$PY_BIN" -u "$ROOT_DIR/scripts/train.py" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --champion-mapping-path "$CHAMPION_MAPPING_PATH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --category "$CATEGORY" \
    --dataset-label "$DATASET_LABEL" \
    --display-name "$display_name" \
    --description "Embedding ablation (priors + eligibility off): $display_name" \
    "$@"
}

# Keep baseline optional because a dedicated all-off baseline already exists.
if [[ "$RUN_BASELINE" == "1" ]]; then
  run_variant "Embeddings ablation: league off, team off" --no-league-team-embeddings
fi

run_variant "Embeddings ablation: league on, team off" --no-team-embeddings
run_variant "Embeddings ablation: league off, team on" --no-league-embeddings
run_variant "Embeddings ablation: league on, team on"

echo "Completed embedding ablation runs."
echo "Experiment index: $OUTPUT_DIR/experiment-index.json"

if [[ "$PUBLISH_DATA" == "1" ]]; then
  "$PY_BIN" -u "$ROOT_DIR/scripts/publish_training_data.py" \
    --index "$OUTPUT_DIR/experiment-index.json" \
    --data-dir "$PUBLISH_DATA_DIR"
  echo "Published to $PUBLISH_DATA_DIR"
fi
