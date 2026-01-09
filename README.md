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
- Training must span multiple patches; `patch` is a required feature.

## Outputs (MVP)
- Model artifact(s)
- Metrics summary (accuracy/loss/top-k)
- Training config metadata for reproducibility

## Configuration (MVP targets)
- Input path (processor output location)
- Output directory
- Train/validation split ratio
- Random seed
- Epochs, batch size, learning rate
- Patch window selection (rolling or explicit list)

## Status
Scaffold only. No training pipeline implemented yet.
