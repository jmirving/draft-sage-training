# TODO — Training Data Additions

## Features (model learns from IDs or direct inputs)

### Team/league
- [x] Team/league categorical embeddings (exp-league-team-features; current best results).
- [x] Decide if team/league features are the baseline for all future experiments. (Decision: do not default; run as individual experiments first.)

### Action/side
- [x] Action/side/event index features (exp-action-side-event; solid baseline).

## Weights (precomputed priors fed as inputs)
- [x] Define common artifact conventions (versioning, patch keys, normalization rules). (Decision: per-patch files, probabilities, mapping by normalized_name; see WEIGHTS_CONVENTIONS.md.)
- [x] Decide the shared data window rules (clean 2025 base, patch-aware recency). (Decision: per-patch weights.)
- [ ] Add config/CLI plumbing for optional weight artifacts.
- [ ] Extend dataset loader to accept optional weight features (no-op when missing).
- [ ] Add basic validation + tests for missing/unknown keys.

### Role weights (per-champion role priors)
- [ ] Confirm role prior schema (per patch, per champion, 5-role distribution, normalization rule). (Expect low per-patch variance.)
- [ ] Decide data source + window (clean 2025 games; DDragon-only mapping).
- [ ] Add artifact generator script to produce `data/role-priors/latest.json`.
- [ ] Update model to consume the role prior feature (keep it simple).
- [ ] Log role prior metadata (path/version) in training metrics output.
- [ ] Decide whether the role priors artifact stays in-repo or is generated externally later.

### Player weights (player champion-pool priors)
- [ ] Add player champion-pool priors per game using the 10 players tied to the game ID.
- [ ] Define recency weighting for player histories (e.g., patch-aware decay; ignore stale picks).
- [ ] Decide player-pool artifact shape (per player, per patch window, normalized distribution).

### Team/league weights (team + region priors)
- [ ] Confirm team/league prior schema (per patch, per team/league, champion distribution).
- [ ] Decide data source + window (clean 2025 games; DDragon-only mapping).
- [ ] Add artifact generator script to produce `data/team-league-priors/latest.json`.
- [ ] Update model to consume team/league priors (keep it simple).

## Experiments + storage
- [ ] Run a clean 2025 experiment (20 epochs) and record results in the shared summary area.
