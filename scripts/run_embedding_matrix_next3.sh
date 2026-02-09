#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-$ROOT_DIR/.venv/bin/python}"

INPUT_DIR="${INPUT_DIR:-}"
if [[ -z "$INPUT_DIR" ]]; then
  for candidate in \
    "$ROOT_DIR/data/prodata" \
    "$ROOT_DIR/../.tmp/prodata-2025-clean" \
    "$ROOT_DIR/../.tmp/prodata-2025" \
    "$ROOT_DIR/../.tmp/training-real/prodata-processed" \
    "$ROOT_DIR/../draft-sage/data/processed"
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
DATASET_LABEL="${DATASET_LABEL:-Clean 2025}"
BASE_DIR="${BASE_DIR:-$ROOT_DIR/../.tmp/embedding-matrix-$(date -u +%Y%m%d_%H%M%S)-ep20-next3}"

mkdir -p "$BASE_DIR/logs"
STATUS_FILE="$BASE_DIR/status.txt"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$STATUS_FILE"
}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

run_one() {
  local name="$1"
  shift
  local out_dir="$BASE_DIR/$name"
  local log_file="$BASE_DIR/logs/$name.log"

  log "START $name"
  (set -x; "$PY_BIN" -u "$ROOT_DIR/scripts/train.py" \
    --output-dir "$out_dir" \
    --display-name "Embeddings: ${name//_/ }" \
    --description "Ablation: ${name//_/ }" \
    --input-dir "$INPUT_DIR" \
    --champion-mapping-path "$CHAMPION_MAPPING_PATH" \
    --epochs "$EPOCHS" \
    --seed "$SEED" \
    --category embedding-matrix \
    --dataset-label "$DATASET_LABEL" \
    "$@") >"$log_file" 2>&1 &

  local pid=$!
  echo "$pid" > "$BASE_DIR/$name.pid"
  log "PID $name=$pid"
}

# Launch three ablations in the background (concurrent).
run_one "league_off_team_off" --no-league-embeddings --no-team-embeddings
run_one "league_on_team_off" --no-team-embeddings
run_one "league_off_team_on" --no-league-embeddings

log "Launched 3 runs under $BASE_DIR"
