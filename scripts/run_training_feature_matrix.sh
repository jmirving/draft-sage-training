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

# Embedding matrix axes.
# Add new embedding axes by listing them here and defining <AXIS>_EMBEDDINGS_VALUES.
# Example: EMBEDDING_AXES=league,team,player and PLAYER_EMBEDDINGS_VALUES=off,on
EMBEDDING_AXES="${EMBEDDING_AXES:-league,team}"

# Per-axis values are read from <UPPERCASE_AXIS>_EMBEDDINGS_VALUES.
# Defaults below preserve the original 3-run ablation.
LEAGUE_EMBEDDINGS_VALUES="${LEAGUE_EMBEDDINGS_VALUES:-off,on}"
TEAM_EMBEDDINGS_VALUES="${TEAM_EMBEDDINGS_VALUES:-off,on}"

# Include the all-on embedding baseline when both bias axes are off (1=yes, 0=no).
# Legacy alias: SKIP_BASELINE_ON_ON (inverse)
INCLUDE_ALL_EMBEDDINGS_BASELINE="${INCLUDE_ALL_EMBEDDINGS_BASELINE:-}"
if [[ -z "$INCLUDE_ALL_EMBEDDINGS_BASELINE" ]]; then
  legacy_skip="${SKIP_BASELINE_ON_ON:-1}"
  if [[ "$legacy_skip" == "1" ]]; then
    INCLUDE_ALL_EMBEDDINGS_BASELINE="0"
  else
    INCLUDE_ALL_EMBEDDINGS_BASELINE="1"
  fi
fi

# Optional bias axes (clear names).
# Legacy aliases:
# - CHAMPION_PRIORS_* => PICK_BAN_PRIOR_BIAS_*
# - ROLE_PRIORS_* => ROLE_DISTRIBUTION_BIAS_*
PICK_BAN_PRIOR_BIAS_VALUES="${PICK_BAN_PRIOR_BIAS_VALUES:-${CHAMPION_PRIORS_VALUES:-off}}"
ROLE_DISTRIBUTION_BIAS_VALUES="${ROLE_DISTRIBUTION_BIAS_VALUES:-${ROLE_PRIORS_VALUES:-off}}"

PICK_BAN_PRIOR_BIAS_DIR="${PICK_BAN_PRIOR_BIAS_DIR:-${CHAMPION_PRIORS_DIR:-$ROOT_DIR/data/weights/champion-priors}}"
PICK_BAN_PRIOR_BIAS_STRENGTH_VALUES="${PICK_BAN_PRIOR_BIAS_STRENGTH_VALUES:-${CHAMPION_PRIORS_STRENGTH_VALUES:-1.0}}"
PICK_BAN_PRIOR_BIAS_TIME_BUCKET_VALUES="${PICK_BAN_PRIOR_BIAS_TIME_BUCKET_VALUES:-${CHAMPION_PRIORS_TIME_BUCKET_VALUES:-1}}"

ROLE_DISTRIBUTION_BIAS_DIR="${ROLE_DISTRIBUTION_BIAS_DIR:-${ROLE_PRIORS_DIR:-$ROOT_DIR/data/weights/role-priors}}"
ROLE_DISTRIBUTION_BIAS_STRENGTH_VALUES="${ROLE_DISTRIBUTION_BIAS_STRENGTH_VALUES:-${ROLE_PRIORS_STRENGTH_VALUES:-1.0}}"

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

axis_to_upper_var_prefix() {
  local axis="$1"
  printf '%s' "$axis" | tr '[:lower:]' '[:upper:]'
}

axis_values_env_name() {
  local axis="$1"
  local upper
  upper="$(axis_to_upper_var_prefix "$axis")"
  printf '%s_EMBEDDINGS_VALUES' "$upper"
}

validate_axis_name() {
  local axis="$1"
  if [[ ! "$axis" =~ ^[a-z][a-z0-9_]*$ ]]; then
    echo "Invalid axis name '$axis'. Use lowercase snake_case (example: league, team, player)." >&2
    exit 1
  fi
}

supports_axis_flag() {
  local axis="$1"
  local help_text="$2"
  local flag="--no-${axis}-embeddings"
  if ! grep -q -- "$flag" <<< "$help_text"; then
    echo "Axis '$axis' is not supported by train.py (missing flag $flag)." >&2
    exit 1
  fi
}

declare -a EMBEDDING_AXES_ARR=()
parse_csv "$EMBEDDING_AXES" EMBEDDING_AXES_ARR
if [[ "${#EMBEDDING_AXES_ARR[@]}" -eq 0 ]]; then
  echo "EMBEDDING_AXES cannot be empty." >&2
  exit 1
fi

TRAIN_HELP="$($PY_BIN "$ROOT_DIR/scripts/train.py" --help 2>&1 || true)"

declare -A AXIS_VALUES_CSV=()
declare -a AXES_LOG_PARTS=()
for axis in "${EMBEDDING_AXES_ARR[@]}"; do
  validate_axis_name "$axis"
  supports_axis_flag "$axis" "$TRAIN_HELP"

  values_env_name="$(axis_values_env_name "$axis")"
  values_csv="${!values_env_name:-off,on}"

  declare -a axis_values=()
  parse_csv "$values_csv" axis_values
  if [[ "${#axis_values[@]}" -eq 0 ]]; then
    echo "Axis '$axis' values cannot be empty (env: $values_env_name)." >&2
    exit 1
  fi
  validate_on_off_values "$values_env_name" "${axis_values[@]}"

  AXIS_VALUES_CSV["$axis"]="$values_csv"
  AXES_LOG_PARTS+=("$axis=[$values_csv]")
done

declare -a PICK_BAN_BIAS_AXIS=()
declare -a ROLE_DIST_BIAS_AXIS=()
declare -a PICK_BAN_BIAS_STRENGTHS=()
declare -a PICK_BAN_BIAS_TIME_BUCKETS=()
declare -a ROLE_DIST_BIAS_STRENGTHS=()

parse_csv "$PICK_BAN_PRIOR_BIAS_VALUES" PICK_BAN_BIAS_AXIS
parse_csv "$ROLE_DISTRIBUTION_BIAS_VALUES" ROLE_DIST_BIAS_AXIS
parse_csv "$PICK_BAN_PRIOR_BIAS_STRENGTH_VALUES" PICK_BAN_BIAS_STRENGTHS
parse_csv "$PICK_BAN_PRIOR_BIAS_TIME_BUCKET_VALUES" PICK_BAN_BIAS_TIME_BUCKETS
parse_csv "$ROLE_DISTRIBUTION_BIAS_STRENGTH_VALUES" ROLE_DIST_BIAS_STRENGTHS

validate_on_off_values "PICK_BAN_PRIOR_BIAS_VALUES" "${PICK_BAN_BIAS_AXIS[@]}"
validate_on_off_values "ROLE_DISTRIBUTION_BIAS_VALUES" "${ROLE_DIST_BIAS_AXIS[@]}"

if [[ "${#PICK_BAN_BIAS_AXIS[@]}" -eq 0 || "${#ROLE_DIST_BIAS_AXIS[@]}" -eq 0 ]]; then
  echo "Bias axes cannot be empty." >&2
  exit 1
fi

if [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "MAX_PARALLEL must be >= 1." >&2
  exit 1
fi

if [[ "$INCLUDE_ALL_EMBEDDINGS_BASELINE" != "0" && "$INCLUDE_ALL_EMBEDDINGS_BASELINE" != "1" ]]; then
  echo "INCLUDE_ALL_EMBEDDINGS_BASELINE must be 0 or 1." >&2
  exit 1
fi

if contains_on "${PICK_BAN_BIAS_AXIS[@]}" && [[ ! -d "$PICK_BAN_PRIOR_BIAS_DIR" ]]; then
  echo "PICK_BAN_PRIOR_BIAS_DIR not found but pick/ban bias axis includes 'on': $PICK_BAN_PRIOR_BIAS_DIR" >&2
  exit 1
fi

if contains_on "${ROLE_DIST_BIAS_AXIS[@]}" && [[ ! -d "$ROLE_DISTRIBUTION_BIAS_DIR" ]]; then
  echo "ROLE_DISTRIBUTION_BIAS_DIR not found but role-distribution bias axis includes 'on': $ROLE_DISTRIBUTION_BIAS_DIR" >&2
  exit 1
fi

if [[ "${#PICK_BAN_BIAS_STRENGTHS[@]}" -eq 0 || "${#PICK_BAN_BIAS_TIME_BUCKETS[@]}" -eq 0 ]]; then
  echo "Pick/ban prior bias strength/time bucket lists cannot be empty." >&2
  exit 1
fi

if [[ "${#ROLE_DIST_BIAS_STRENGTHS[@]}" -eq 0 ]]; then
  echo "Role-distribution bias strength list cannot be empty." >&2
  exit 1
fi

declare -a ACTIVE_PIDS=()
declare -a ACTIVE_NAMES=()
declare -a FAILED_RUNS=()
TOTAL_RUNS=0

declare -a CURRENT_EMBED_VALUES=()
declare -a CURRENT_ARGS=()

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
      --display-name "Feature matrix: ${name//_/ }" \
      --description "Feature matrix ablation: ${name//_/ }" \
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

all_embeddings_on() {
  local value
  for value in "${CURRENT_EMBED_VALUES[@]}"; do
    if [[ "$value" != "on" ]]; then
      return 1
    fi
  done
  return 0
}

build_embedding_name() {
  local name=""
  local i
  for i in "${!EMBEDDING_AXES_ARR[@]}"; do
    name+="${EMBEDDING_AXES_ARR[$i]}_${CURRENT_EMBED_VALUES[$i]}_"
  done
  name="${name%_}"
  printf '%s' "$name"
}

run_leaf_for_current_embeddings() {
  local pick_ban_bias="$1"
  local role_dist_bias="$2"

  if [[ "$INCLUDE_ALL_EMBEDDINGS_BASELINE" != "1" && "$pick_ban_bias" == "off" && "$role_dist_bias" == "off" ]]; then
    if all_embeddings_on; then
      return
    fi
  fi

  local -a pick_ban_strength_loop=("na")
  local -a pick_ban_bucket_loop=("na")
  local -a role_dist_strength_loop=("na")

  if [[ "$pick_ban_bias" == "on" ]]; then
    pick_ban_strength_loop=("${PICK_BAN_BIAS_STRENGTHS[@]}")
    pick_ban_bucket_loop=("${PICK_BAN_BIAS_TIME_BUCKETS[@]}")
  fi

  if [[ "$role_dist_bias" == "on" ]]; then
    role_dist_strength_loop=("${ROLE_DIST_BIAS_STRENGTHS[@]}")
  fi

  local pick_ban_strength
  local pick_ban_bucket
  local role_dist_strength
  for pick_ban_strength in "${pick_ban_strength_loop[@]}"; do
    for pick_ban_bucket in "${pick_ban_bucket_loop[@]}"; do
      for role_dist_strength in "${role_dist_strength_loop[@]}"; do
        local run_name
        run_name="$(build_embedding_name)_pbbias_${pick_ban_bias}_roledist_${role_dist_bias}"
        local -a run_args=("${CURRENT_ARGS[@]}")

        if [[ "$pick_ban_bias" == "on" ]]; then
          run_args+=(--champion-priors-dir "$PICK_BAN_PRIOR_BIAS_DIR")
          run_args+=(--champion-priors-strength "$pick_ban_strength")
          run_args+=(--champion-priors-time-buckets "$pick_ban_bucket")
          run_name+="_pbstr_$(slug "$pick_ban_strength")_pbbkt_$(slug "$pick_ban_bucket")"
        fi

        if [[ "$role_dist_bias" == "on" ]]; then
          run_args+=(--role-priors-dir "$ROLE_DISTRIBUTION_BIAS_DIR")
          run_args+=(--role-priors-strength "$role_dist_strength")
          run_name+="_rdbstr_$(slug "$role_dist_strength")"
        fi

        run_one "$run_name" "${run_args[@]}"
        while [[ "${#ACTIVE_PIDS[@]}" -ge "$MAX_PARALLEL" ]]; do
          wait_oldest
        done
      done
    done
  done
}

walk_embedding_axes() {
  local idx="$1"

  if [[ "$idx" -ge "${#EMBEDDING_AXES_ARR[@]}" ]]; then
    local pick_ban_bias
    local role_dist_bias
    for pick_ban_bias in "${PICK_BAN_BIAS_AXIS[@]}"; do
      for role_dist_bias in "${ROLE_DIST_BIAS_AXIS[@]}"; do
        run_leaf_for_current_embeddings "$pick_ban_bias" "$role_dist_bias"
      done
    done
    return
  fi

  local axis="${EMBEDDING_AXES_ARR[$idx]}"
  local values_csv="${AXIS_VALUES_CSV[$axis]}"
  local -a axis_values=()
  parse_csv "$values_csv" axis_values

  local value
  for value in "${axis_values[@]}"; do
    CURRENT_EMBED_VALUES+=("$value")

    if [[ "$value" == "off" ]]; then
      CURRENT_ARGS+=("--no-${axis}-embeddings")
    fi

    walk_embedding_axes $((idx + 1))

    if [[ "$value" == "off" ]]; then
      unset 'CURRENT_ARGS[${#CURRENT_ARGS[@]}-1]'
    fi
    unset 'CURRENT_EMBED_VALUES[${#CURRENT_EMBED_VALUES[@]}-1]'
  done
}

log "BASE_DIR=$BASE_DIR"
log "MAX_PARALLEL=$MAX_PARALLEL DRY_RUN=$DRY_RUN"
log "EMBEDDING_AXES=[$EMBEDDING_AXES]"
log "INCLUDE_ALL_EMBEDDINGS_BASELINE=$INCLUDE_ALL_EMBEDDINGS_BASELINE"
log "AXES ${AXES_LOG_PARTS[*]} pick_ban_prior_bias=[$PICK_BAN_PRIOR_BIAS_VALUES] role_distribution_bias=[$ROLE_DISTRIBUTION_BIAS_VALUES]"

walk_embedding_axes 0

if [[ "$DRY_RUN" != "1" ]]; then
  wait_for_all
fi

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  log "Completed with failures (${#FAILED_RUNS[@]}/$TOTAL_RUNS): ${FAILED_RUNS[*]}"
  exit 1
fi

log "Completed $TOTAL_RUNS run(s) under $BASE_DIR"
