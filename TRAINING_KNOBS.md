# Training Knobs Reference

This document is the canonical list of knobs for training runs and matrix sweeps.
Use it when defining experiment variants and mapping controls into the training UI.

## 1. Matrix Runner Knobs (`scripts/run_training_feature_matrix.sh`)

### Core execution

| Knob | Default | Purpose |
|---|---|---|
| `INPUT_DIR` | auto-resolved | Processed prodata source directory. |
| `CHAMPION_MAPPING_PATH` | auto-resolved | Champion mapping artifact path. |
| `EPOCHS` | `20` | Epochs per run. |
| `SEED` | `42` | Random seed per run. |
| `DATASET_LABEL` | `Clean 2025` | Label written to run metadata. |
| `CATEGORY` | `embedding-matrix` | Summary category for UI grouping. |
| `BASE_DIR` | timestamped `.tmp` path | Parent directory for matrix outputs/logs. |
| `MAX_PARALLEL` | `3` | Max concurrent training jobs. |
| `DRY_RUN` | `0` | `1` prints matrix and args without launching training. |

### Embedding axes (generic)

| Knob | Default | Purpose |
|---|---|---|
| `EMBEDDING_AXES` | `league,team` | Comma-separated embedding axes to sweep. |
| `<AXIS>_EMBEDDINGS_VALUES` | `off,on` | Per-axis values (`on/off`) for each axis in `EMBEDDING_AXES`. |
| `INCLUDE_ALL_EMBEDDINGS_BASELINE` | `0` | Include run where all embeddings are `on` when both bias axes are `off`. |

Notes:
- Axis names must be lowercase snake_case (example: `league`, `team`, `player`).
- For each axis, training CLI must support `--no-<axis>-embeddings`.
- Example future axis:
  - `EMBEDDING_AXES=league,team,player`
  - `PLAYER_EMBEDDINGS_VALUES=off,on`

### Bias axes + weight sweeps

| Knob | Default | Purpose |
|---|---|---|
| `PICK_BAN_PRIOR_BIAS_VALUES` | `off` | Include pick/ban frequency bias axis (`off/on`). |
| `ROLE_DISTRIBUTION_BIAS_VALUES` | `off` | Include role-distribution bias axis (`off/on`). |
| `PICK_BAN_PRIOR_BIAS_DIR` | `data/weights/champion-priors` | Pick/ban bias artifact directory. |
| `PICK_BAN_PRIOR_BIAS_STRENGTH_VALUES` | `1.0` | Comma-separated pick/ban bias strengths. |
| `PICK_BAN_PRIOR_BIAS_TIME_BUCKET_VALUES` | `1` | Comma-separated pick/ban bias time-bucket values. |
| `ROLE_DISTRIBUTION_BIAS_DIR` | `data/weights/role-priors` | Role-distribution bias artifact directory. |
| `ROLE_DISTRIBUTION_BIAS_STRENGTH_VALUES` | `1.0` | Comma-separated role-distribution bias strengths. |

Legacy aliases still supported in the matrix runner:
- `SKIP_BASELINE_ON_ON` (inverse of `INCLUDE_ALL_EMBEDDINGS_BASELINE`)
- `CHAMPION_PRIORS_*` (maps to `PICK_BAN_PRIOR_BIAS_*`)
- `ROLE_PRIORS_*` (maps to `ROLE_DISTRIBUTION_BIAS_*`)

Naming note:
- Matrix runner uses intent-first names (`pick/ban prior bias`, `role-distribution bias`).
- Training CLI keeps historical flag names (`--champion-priors-*`, `--role-priors-*`) for compatibility.

### Common matrix commands

```bash
# 1) Default 3-run embedding ablation (legacy behavior)
./scripts/run_training_feature_matrix.sh

# 2) Full 2x2 embedding matrix (includes all-on baseline)
INCLUDE_ALL_EMBEDDINGS_BASELINE=1 ./scripts/run_training_feature_matrix.sh

# 3) Include bias on/off axes
PICK_BAN_PRIOR_BIAS_VALUES=off,on ROLE_DISTRIBUTION_BIAS_VALUES=off,on \
  ./scripts/run_training_feature_matrix.sh

# 4) Bias strength sweep (both bias axes enabled)
PICK_BAN_PRIOR_BIAS_VALUES=on ROLE_DISTRIBUTION_BIAS_VALUES=on \
PICK_BAN_PRIOR_BIAS_STRENGTH_VALUES=0.5,1.0 ROLE_DISTRIBUTION_BIAS_STRENGTH_VALUES=0.5,1.0 \
  ./scripts/run_training_feature_matrix.sh

# 5) Dry-run the resolved grid without training
DRY_RUN=1 PICK_BAN_PRIOR_BIAS_VALUES=off,on ROLE_DISTRIBUTION_BIAS_VALUES=off,on \
  ./scripts/run_training_feature_matrix.sh
```

## 2. Training CLI Knobs (`scripts/train.py`)

### Inputs and artifacts
- `--input-dir`
- `--output-dir`
- `--champion-mapping-path`
- `--champion-eligibility-path`
- `--champion-priors-dir`
- `--role-priors-dir`

### Priors behavior
- `--champion-priors-strength`
- `--champion-priors-time-buckets`
- `--role-priors-strength`

### Embedding toggles
- `--no-league-team-embeddings`
- `--no-league-embeddings`
- `--no-team-embeddings`
- Pattern for future axes: `--no-<axis>-embeddings`

### Training and split
- `--train-split`
- `--val-split`
- `--test-split`
- `--split-strategy` (`seriesid`, `gameid`, `random`)
- `--seed`
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--patch-window`
- `--patch` (repeatable)

### Metadata (UI-facing)
- `--category`
- `--display-name`
- `--description`
- `--dataset-label`

### Index/publish controls
- `--no-index-update`
- `--publish-data`
- `--publish-on-start`
- `--publish-on-finish`
- `--publish-data-dir`
- `--publish-index` (repeatable)
- `--publish-commit`
- `--publish-push`
- `--inspection-keep`
- `--log-level`

## 3. UI-Relevant Metadata Produced per Run

Primary fields for the training UI come from:
- `summary.json`: `category`, `display_name`, `description`, `dataset`, `metrics`, `status`, `progress`, `inspection_status`, `paths`
- `metrics.json`: `feature_set`, `test_accuracy`, `best_val_loss`, sample counts, priors metadata
- `config.json`: full resolved configuration for exact run reproducibility

Use this knobs file + `config.json` as the source for control labels and diff views in the UI.
