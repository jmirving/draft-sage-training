#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/.venv}"
PIP_BIN="$VENV_PATH/bin/pip"
PY_BIN="$VENV_PATH/bin/python"

if [[ ! -x "$PY_BIN" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi

"$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"

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
  DDRAGON_BUILDER_ROOT="${DDRAGON_BUILDER_ROOT:-$ROOT_DIR/../lol-ddragon-context-artifact-builder}"
  DDRAGON_SNAPSHOT_BASE="${DDRAGON_SNAPSHOT_BASE:-$ROOT_DIR/../lol-ddragon-snapshot-cron/data/ddragon/extracted}"
  DDRAGON_SNAPSHOT_LOCALE="${DDRAGON_SNAPSHOT_LOCALE:-en_US}"
  BUILDER_OUTPUT_PATH="$DDRAGON_BUILDER_ROOT/data/ddragon/artifacts/champion-mapping/latest.json"
  if [[ -d "$DDRAGON_BUILDER_ROOT" && -d "$DDRAGON_SNAPSHOT_BASE" ]]; then
    SNAPSHOT_VERSION="$(find "$DDRAGON_SNAPSHOT_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -V | tail -n 1)"
    if [[ -n "$SNAPSHOT_VERSION" ]]; then
      echo "Champion mapping missing; building latest from DDragon snapshot ($SNAPSHOT_VERSION)."
      (
        cd "$DDRAGON_BUILDER_ROOT"
        GRADLE_USER_HOME="${GRADLE_USER_HOME:-/tmp/.gradle-clean}" gradle_safe run \
          --args="--snapshot-base-uri=$DDRAGON_SNAPSHOT_BASE --snapshot-version=$SNAPSHOT_VERSION --snapshot-locale=$DDRAGON_SNAPSHOT_LOCALE --artifacts-base-uri=$DDRAGON_BUILDER_ROOT/data --artifact-version=latest"
      )
      if [[ -f "$BUILDER_OUTPUT_PATH" ]]; then
        CHAMPION_MAPPING_PATH="$BUILDER_OUTPUT_PATH"
      fi
    fi
  fi
fi

if [[ -z "$CHAMPION_MAPPING_PATH" ]]; then
  echo "No champion mapping found. Set CHAMPION_MAPPING_PATH to a mapping JSON." >&2
  echo "Checked snapshot base: ${DDRAGON_SNAPSHOT_BASE:-<unset>}" >&2
  echo "Checked builder root: ${DDRAGON_BUILDER_ROOT:-<unset>}" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/../.tmp/training-autopublish}"
PUBLISH_DATA_DIR="${PUBLISH_DATA_DIR:-$ROOT_DIR/../draft-sage-training-data/public/training}"
EPOCHS="${EPOCHS:-1}"

if [[ ! -d "$PUBLISH_DATA_DIR" ]]; then
  echo "Publish data dir not found: $PUBLISH_DATA_DIR" >&2
  exit 1
fi

DATA_REPO_ROOT="$ROOT_DIR/../draft-sage-training-data"
if [[ "${ALLOW_DIRTY_DATA_REPO:-}" != "1" ]]; then
  DIRTY_STATE="$(cd "$DATA_REPO_ROOT" && git_local status --porcelain)"
  if [[ -n "$DIRTY_STATE" ]]; then
    echo "Data repo has uncommitted changes. Set ALLOW_DIRTY_DATA_REPO=1 to bypass." >&2
    exit 1
  fi
fi

EXTRA_ARGS=("$@")
if [[ ! " ${EXTRA_ARGS[*]} " =~ --epochs ]]; then
  EXTRA_ARGS+=(--epochs "$EPOCHS")
fi

"$PY_BIN" "$ROOT_DIR/scripts/train.py" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --champion-mapping-path "$CHAMPION_MAPPING_PATH" \
  --publish-data \
  --publish-data-dir "$PUBLISH_DATA_DIR" \
  "${EXTRA_ARGS[@]}"
