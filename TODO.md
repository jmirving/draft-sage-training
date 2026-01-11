# TODO — Draft Sage Training

## Immediate next items (weights so new experiments are possible)
- [ ] Decide weight artifact conventions (JSON schema, versioning, normalization rules, required metadata).
- [ ] Decide the shared data window policy for weights (clean 2025 vs patch-aware recency).
- [ ] Add CLI/config plumbing for optional weights (e.g., `--weights-path`).
- [ ] Load weights and align to champion indices; default to zero when missing.
- [ ] Add validation (unknown champions, missing values, shape checks) + tests.
- [ ] Record weight metadata (path/version/window) in `summary.json`.

## Role priors (if/when weights are in place)
- [ ] Confirm role prior schema (per patch, per champion, 5-role distribution, normalization rule).
- [ ] Decide data source + window (clean 2025 games; DDragon-only mapping).
- [ ] Add artifact generator script to produce `data/role-priors/latest.json`.
- [ ] Update model to consume the role prior feature (keep it simple).
- [ ] Log role prior metadata in training summaries.

## Team/league priors (later)
- [ ] Confirm team/league prior schema (per patch, per team/league, champion distribution).
- [ ] Decide data source + window (clean 2025 games; DDragon-only mapping).
- [ ] Add artifact generator script to produce `data/team-league-priors/latest.json`.
- [ ] Update model to consume team/league priors.

## Experiments (after weights/priors land)
- [ ] Run weight-enabled baseline and record results in the shared index.
- [ ] Compare best runs by category in the training UI.
