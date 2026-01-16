# Weight Artifact Conventions (v1)

## Purpose
Define a simple, inspectable weight artifact format so per-patch weights can be
generated, compared, and consumed consistently.

## Weight meanings (naming clarity)
- `champion-priors`: Per-patch **pick/ban frequency priors** for champions. These are
  distributions over champions (sum-to-1) and are added as a logit bias for
  both picks and bans.
- `role-priors`: Per-patch **champion role distributions** (Top/Jungle/Mid/Bot/Sup).
  These are used to bias picks toward unmet team roles and are only applied to
  pick actions (ban actions receive a zero role bias).

## Storage layout
Per-patch files, one file per patch:

```
data/weights/<weight_type>/<patch>.json
```

Examples:
- `data/weights/champion-priors/15.1.json`
- `data/weights/role-priors/15.1.json`

## Schema (v1)
```json
{
  "schema_version": "1.0",
  "weight_type": "champion-priors",
  "patch": "15.1",
  "generated_at": "2026-01-11T20:15:00Z",
  "normalization": "sum-to-1",
  "weights": {
    "ahri": 0.0062,
    "wukong": 0.0041
  },
  "meta": {
    "source": "clean-2025",
    "window": "per-patch"
  }
}
```

Required fields:
- `schema_version`: `"1.0"`
- `weight_type`: string (e.g., `champion-priors`, `role-priors`)
- `patch`: patch string as used in the training data (e.g., `"15.1"`)
- `normalization`: `"sum-to-1"`
- `weights`: object mapping `normalized_name` -> probability

Optional fields:
- `generated_at`: ISO-8601 UTC timestamp
- `meta`: free-form metadata (source, window, notes)

## Normalization rules
- Each file is normalized per patch (`sum(weights) == 1.0`).
- Missing champions are treated as weight `0.0`.
- Keys must use `normalized_name` from the DDragon mapping.

## Validation expectations
- Unknown champion names are rejected.
- Negative weights are rejected.
- Weights must sum to 1.0 within a small tolerance (e.g., ±1e-6).
