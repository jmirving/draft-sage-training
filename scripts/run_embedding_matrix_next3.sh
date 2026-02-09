#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[deprecated] run_embedding_matrix_next3.sh -> run_training_feature_matrix.sh" >&2
exec "$SCRIPT_DIR/run_training_feature_matrix.sh" "$@"
