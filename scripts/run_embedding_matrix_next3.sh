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
CATEGORY="${CATEGORY:-embedding-matrix}"
BASE_DIR="${BASE_DIR:-$ROOT_DIR/../.tmp/embedding-matrix-$(date -u +%Y%m%d_%H%M%S)-ep${EPOCHS}}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
DRY_RUN="${DRY_RUN:-0}"

# Matrix axes
# Defaults preserve the old "next3" behavior (3 runs: off/off, on/off, off/on).
LEAGUE_EMBEDDINGS_VALUES="${LEAGUE_EMBEDDINGS_VALUES:-off,on}"
TEAM_EMBEDDINGS_VALUES="${TEAM_EMBEDDINGS_VALUES:-off,on}"
CHAMPION_PRIORS_VALUES="${CHAMPION_PRIORS_VALUES:-off}"
ROLE_PRIORS_VALUES="${ROLE_PRIORS_VALUES:-off}"
SKIP_BASELINE_ON_ON="${SKIP_BASELINE_ON_ON:-1}"

# Optional weight sweeps (used when corresponding axis includes "on")
CHAMPION_PRIORS_DIR="${CHAMPION_PRIORS_DIR:-$ROOT_DIR/data/weights/champion-priors}"
CHAMPION_PRIORS_STRENGTH_VALUES="${CHAMPION_PRIORS_STRENGTH_VALUES:-1.0}"
CHAMPION_PRIORS_TIME_BUCKET_VALUES="${CHAMPION_PRIORS_TIME_BUCKET_VALUES:-1}"
ROLE_PRIORS_DIR="${ROLE_PRIORS_DIR:-$ROOT_DIR/data/weights/role-priors}"
ROLE_PRIORS_STRENGTH_VALUES="${ROLE_PRIORS_STRENGTH_VALUES:-1.0}"

mkdir -p "$BASE_DIR/logs"
STATUS_FILE="$BASE_DIR/status.txt"

log() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$STATUS_FILE"
}

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

parse_csv() {
  local input="$1"
  local -n output_ref="$2"
  output_ref=()
  local raw=()
  IFS=',' read -r -a raw <<< "$input"
  local item
  for item in "${raw[@]}"; do
    item="$(trim "$item")"
    if [[ -n "$item" ]]; then
      output_ref+=("$item")
    fi
  done
}

validate_on_off_values() {
  local axis_name="$1"
  shift
  local value
  for value in "$@"; do
    case "$value" in
      on|off) ;;
      *)
        echo "Invalid value for $axis_name: '$value' (expected on/off)" >&2
        exit 1
        ;;
    esac
  done
}

contains_on() {
  local value
  for value in "$@"; do
    if [[ "$value" == "on" ]]; then
      return 0
    fi
  done
  return 1
}

slug() {
  local value="$1"
  value="$(trim "$value")"
  value="${value//./p}"
  value="${value//[^A-Za-z0-9_-]/_}"
  printf '%s' "$value"
}

declare -a LEAGUE_VALUES=()
declare -a TEAM_VALUES=()
declare -a CHAMPION_PRIOR_AXIS=()
declare -a ROLE_PRIOR_AXIS=()
declare -a CHAMPION_PRIOR_STRENGTHS=()
declare -a CHAMPION_PRIOR_TIME_BUCKETS=()
declare -a ROLE_PRIOR_STRENGTHS=()

parse_csv "$LEAGUE_EMBEDDINGS_VALUES" LEAGUE_VALUES
parse_csv "$TEAM_EMBEDDINGS_VALUES" TEAM_VALUES
parse_csv "$CHAMPION_PRIORS_VALUES" CHAMPION_PRIOR_AXIS
parse_csv "$ROLE_PRIORS_VALUES" ROLE_PRIOR_AXIS
parse_csv "$CHAMPION_PRIORS_STRENGTH_VALUES" CHAMPION_PRIOR_STRENGTHS
parse_csv "$CHAMPION_PRIORS_TIME_BUCKET_VALUES" CHAMPION_PRIOR_TIME_BUCKETS
parse_csv "$ROLE_PRIORS_STRENGTH_VALUES" ROLE_PRIOR_STRENGTHS

validate_on_off_values "LEAGUE_EMBEDDINGS_VALUES" "${LEAGUE_VALUES[@]}"
validate_on_off_values "TEAM_EMBEDDINGS_VALUES" "${TEAM_VALUES[@]}"
validate_on_off_values "CHAMPION_PRIORS_VALUES" "${CHAMPION_PRIOR_AXIS[@]}"
validate_on_off_values "ROLE_PRIORS_VALUES" "${ROLE_PRIOR_AXIS[@]}"

if [[ "${#LEAGUE_VALUES[@]}" -eq 0 || "${#TEAM_VALUES[@]}" -eq 0 ]]; then
  echo "Matrix axes for league/team embeddings cannot be empty." >&2
  exit 1
fi

if [[ "${#CHAMPION_PRIOR_AXIS[@]}" -eq 0 || "${#ROLE_PRIOR_AXIS[@]}" -eq 0 ]]; then
  echo "Matrix axes for champion/role priors cannot be empty." >&2
  exit 1
fi

if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "MAX_PARALLEL must be >= 1." >&2
  exit 1
fi

if contains_on "${CHAMPION_PRIOR_AXIS[@]}" && [[ ! -d "$CHAMPION_PRIORS_DIR" ]]; then
  echo "CHAMPION_PRIORS_DIR not found but champion priors axis includes 'on': $CHAMPION_PRIORS_DIR" >&2
  exit 1
fi

if contains_on "${ROLE_PRIOR_AXIS[@]}" && [[ ! -d "$ROLE_PRIORS_DIR" ]]; then
  echo "ROLE_PRIORS_DIR not found but role priors axis includes 'on': $ROLE_PRIORS_DIR" >&2
  exit 1
fi

if [[ "${#CHAMPION_PRIOR_STRENGTHS[@]}" -eq 0 || "${#CHAMPION_PRIOR_TIME_BUCKETS[@]}" -eq 0 ]]; then
  echo "Champion priors strength/time bucket lists cannot be empty." >&2
  exit 1
fi

if [[ "${#ROLE_PRIOR_STRENGTHS[@]}" -eq 0 ]]; then
  echo "Role priors strength list cannot be empty." >&2
  exit 1
fi

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_NAMES=()
declare -a FAILED_RUNS=()
TOTAL_RUNS=0

run_one() {
  local name="$1"
  shift
  local out_dir="$BASE_DIR/$name"
  local log_file="$BASE_DIR/logs/$name.log"

  log "START $name"
  TOTAL_RUNS=$((TOTAL_RUNS + 1))
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN $name args: $*"
    return
  fi

  (
    set -x
    "$PY_BIN" -u "$ROOT_DIR/scripts/train.py" \
      --output-dir "$out_dir" \
      --display-name "Embeddings: ${name//_/ }" \
      --description "Ablation: ${name//_/ }" \
      --input-dir "$INPUT_DIR" \
      --champion-mapping-path "$CHAMPION_MAPPING_PATH" \
      --epochs "$EPOCHS" \
      --seed "$SEED" \
      --category "$CATEGORY" \
      --dataset-label "$DATASET_LABEL" \
      "$@"
  ) >"$log_file" 2>&1 &

  local pid=$!
  echo "$pid" > "$BASE_DIR/$name.pid"
  ACTIVE_PIDS+=("$pid")
  ACTIVE_NAMES+=("$name")
  log "PID $name=$pid"
}

wait_oldest() {
  local pid="${ACTIVE_PIDS[0]}"
  local name="${ACTIVE_NAMES[0]}"
  if wait "$pid"; then
    log "DONE $name"
  else
    local code=$?
    FAILED_RUNS+=("$name:$code")
    log "FAILED $name exit=$code"
  fi
  ACTIVE_PIDS=("${ACTIVE_PIDS[@]:1}")
  ACTIVE_NAMES=("${ACTIVE_NAMES[@]:1}")
}

wait_for_all() {
  while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    wait_oldest
  done
}

champion_enabled_label() {
  local value="$1"
  if [[ "$value" == "on" ]]; then
    printf 'cp_on'
  else
    printf 'cp_off'
  fi
}

role_enabled_label() {
  local value="$1"
  if [[ "$value" == "on" ]]; then
    printf 'rp_on'
  else
    printf 'rp_off'
  fi
}

log "BASE_DIR=$BASE_DIR"
log "MAX_PARALLEL=$MAX_PARALLEL DRY_RUN=$DRY_RUN"
log "AXES league=[$LEAGUE_EMBEDDINGS_VALUES] team=[$TEAM_EMBEDDINGS_VALUES] champion_priors=[$CHAMPION_PRIORS_VALUES] role_priors=[$ROLE_PRIORS_VALUES]"

for league in "${LEAGUE_VALUES[@]}"; do
  for team in "${TEAM_VALUES[@]}"; do
    for champion_priors in "${CHAMPION_PRIOR_AXIS[@]}"; do
      for role_priors in "${ROLE_PRIOR_AXIS[@]}"; do
        if [[ "$SKIP_BASELINE_ON_ON" == "1" && "$league" == "on" && "$team" == "on" && "$champion_priors" == "off" && "$role_priors" == "off" ]]; then
          continue
        fi

        declare -a champion_strength_loop=("na")
        declare -a champion_bucket_loop=("na")
        declare -a role_strength_loop=("na")

        if [[ "$champion_priors" == "on" ]]; then
          champion_strength_loop=("${CHAMPION_PRIOR_STRENGTHS[@]}")
          champion_bucket_loop=("${CHAMPION_PRIOR_TIME_BUCKETS[@]}")
        fi

        if [[ "$role_priors" == "on" ]]; then
          role_strength_loop=("${ROLE_PRIOR_STRENGTHS[@]}")
        fi

        for champion_strength in "${champion_strength_loop[@]}"; do
          for champion_bucket in "${champion_bucket_loop[@]}"; do
            for role_strength in "${role_strength_loop[@]}"; do
              name="league_${league}_team_${team}_$(champion_enabled_label "$champion_priors")_$(role_enabled_label "$role_priors")"
              args=()

              if [[ "$league" == "off" ]]; then
                args+=(--no-league-embeddings)
              fi
              if [[ "$team" == "off" ]]; then
                args+=(--no-team-embeddings)
              fi

              if [[ "$champion_priors" == "on" ]]; then
                args+=(--champion-priors-dir "$CHAMPION_PRIORS_DIR")
                args+=(--champion-priors-strength "$champion_strength")
                args+=(--champion-priors-time-buckets "$champion_bucket")
                name+="_cps_$(slug "$champion_strength")_cpb_$(slug "$champion_bucket")"
              fi

              if [[ "$role_priors" == "on" ]]; then
                args+=(--role-priors-dir "$ROLE_PRIORS_DIR")
                args+=(--role-priors-strength "$role_strength")
                name+="_rps_$(slug "$role_strength")"
              fi

              run_one "$name" "${args[@]}"
              while [[ "${#ACTIVE_PIDS[@]}" -ge "$MAX_PARALLEL" ]]; do
                wait_oldest
              done
            done
          done
        done
      done
    done
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  wait_for_all
fi

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  log "Completed with failures (${#FAILED_RUNS[@]}/$TOTAL_RUNS): ${FAILED_RUNS[*]}"
  exit 1
fi

log "Completed $TOTAL_RUNS run(s) under $BASE_DIR"
