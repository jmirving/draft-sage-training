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
| `DRAFT_FREQUENCY_BIAS_VALUES` | `off` | Include draft-frequency bias axis (`off/on`). |
| `ROLE_DISTRIBUTION_BIAS_VALUES` | `off` | Include role-distribution bias axis (`off/on`). |
| `DRAFT_FREQUENCY_TIME_AWARE_VALUES` | `off` | Explicitly enable time-aware draft-frequency priors (`off/on`), only when draft-frequency bias is `on`. |
| `DRAFT_FREQUENCY_BIAS_DIR` | `data/weights/champion-priors` | Draft-frequency bias artifact directory. |
| `DRAFT_FREQUENCY_BIAS_STRENGTH_VALUES` | `1.0` | Comma-separated draft-frequency bias strengths. |
| `DRAFT_FREQUENCY_BIAS_TIME_BUCKET_VALUES` | `2` | Comma-separated time-bucket values used when `DRAFT_FREQUENCY_TIME_AWARE_VALUES=on` (must be integers `>1`). |
| `ROLE_DISTRIBUTION_BIAS_DIR` | `data/weights/role-priors` | Role-distribution bias artifact directory. |
| `ROLE_DISTRIBUTION_BIAS_STRENGTH_VALUES` | `1.0` | Comma-separated role-distribution bias strengths. |

Notes:
- `DRAFT_FREQUENCY_TIME_AWARE_VALUES=off` forces patch-only draft-frequency priors (`--champion-priors-time-buckets 1`).
- If `DRAFT_FREQUENCY_TIME_AWARE_VALUES` includes `on`, `DRAFT_FREQUENCY_BIAS_VALUES` must also include `on`.

Naming note:
- Matrix runner uses intent-first names (`draft-frequency bias`, `role-distribution bias`).
- Training CLI uses `--champion-priors-*` and `--role-priors-*` flags internally.

### Common matrix commands

```bash
# 1) Default 3-run embedding ablation (legacy behavior)
./scripts/run_training_feature_matrix.sh

# 2) Full 2x2 embedding matrix (includes all-on baseline)
INCLUDE_ALL_EMBEDDINGS_BASELINE=1 ./scripts/run_training_feature_matrix.sh

# 3) Include bias on/off axes
DRAFT_FREQUENCY_BIAS_VALUES=off,on ROLE_DISTRIBUTION_BIAS_VALUES=off,on \
  ./scripts/run_training_feature_matrix.sh

# 4) Bias strength sweep (both bias axes enabled)
DRAFT_FREQUENCY_BIAS_VALUES=on ROLE_DISTRIBUTION_BIAS_VALUES=on \
DRAFT_FREQUENCY_BIAS_STRENGTH_VALUES=0.5,1.0 ROLE_DISTRIBUTION_BIAS_STRENGTH_VALUES=0.5,1.0 \
  ./scripts/run_training_feature_matrix.sh

# 5) Time-aware draft-frequency sweep (patch split buckets)
DRAFT_FREQUENCY_BIAS_VALUES=on DRAFT_FREQUENCY_TIME_AWARE_VALUES=off,on \
DRAFT_FREQUENCY_BIAS_TIME_BUCKET_VALUES=2,3 \
  ./scripts/run_training_feature_matrix.sh

# 6) Dry-run the resolved grid without training
DRY_RUN=1 DRAFT_FREQUENCY_BIAS_VALUES=off,on ROLE_DISTRIBUTION_BIAS_VALUES=off,on \
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
