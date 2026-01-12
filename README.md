# draft-sage-training

Training pipelines and experiment tracking for DraftSage models.

## Scope
- Train models using processed Oracle's Elixir data from `lol-pro-data-processor`
- Own dataset aggregation and feature engineering for training
- Record training outputs (model artifacts + metrics + config metadata)

## Out of scope
- Raw data acquisition
- Pro data processing/normalization
- UI or inference serving

## Inputs
- CSV outputs from `lol-pro-data-processor`:
  - `all`, `players`, `teams`
- DDragon champion mapping artifact (`normalized_name,name,id,key`), default:
  `data/ddragon/artifacts/champion-mapping/latest.json`.
- Training must span multiple patches; `patch` is a required feature.

## Outputs (MVP)
- Model artifact(s)
- Metrics summary (accuracy/loss/top-k)
- Training config metadata for reproducibility
Default layout: `<output-dir>/<runId>/{model.pth,metrics.json,config.json}`

## Configuration (MVP targets)
- Input path (processor output location)
- Output directory
- Train/validation split ratio
- Random seed
- Epochs, batch size, learning rate
- Patch window selection (rolling or explicit list)

## Run (scaffold)
```
python scripts/train.py --input-dir /path/to/prodata --output-dir ./artifacts \
  --champion-mapping-path /path/to/champion-mapping.json
```

Optional: add per-patch champion priors as a logit bias:
```
python scripts/train.py --input-dir /path/to/prodata --output-dir ./artifacts \
  --champion-mapping-path /path/to/champion-mapping.json \
  --champion-priors-dir data/weights/champion-priors \
  --champion-priors-strength 1.0
```
Add `--no-league-team-embeddings` to disable league/team embeddings for a run.

## Experiment index helper
Training now writes `summary.json` and updates `experiment-index.json`
automatically (disable with `--no-index-update`). If you need to backfill or
rebuild the index, use:

```
python scripts/update_experiment_index.py \
  --output-dir /path/to/output \
  --summary-path /path/to/output/<runId>/summary.json
```

## Combine multiple experiment indexes
To view multiple runs at once in the UI, combine their index files:

```
python scripts/build_combined_index.py \
  --index /path/to/experiment-index.json \
  --index /path/to/another/experiment-index.json \
  --output /path/to/combined/experiment-index.json \
  --root /path/to/server/root
```

## Status
Scaffold only. No training pipeline implemented yet.
