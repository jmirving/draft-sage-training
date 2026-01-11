# Test Cases — Clean 2025 + DDragon Mapping (20 Epochs)

**Dataset:** clean 2025 games (`/home/jirving/projects/lol/.tmp/prodata-2025-clean`)
**Mapping:** DDragon-only (`/home/jirving/projects/lol/lol-ddragon-snapshot-cron/data/ddragon/artifacts/champion-mapping/latest.json`)
**Summary source:** `/home/jirving/projects/lol/.tmp/training-clean-2025-ep20/summary.csv`

| Experiment | Run ID | Epochs | Train | Val | Test | Best Val Loss | Test Loss | Test Acc | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp-action-side-event | 20260111_005752 | 20 | 142224 | 17778 | 17778 | 3.19035 | 3.22030 | 0.18292 | Baseline features |
| exp-ban-pick-heads | 20260111_012009 | 20 | 142224 | 17778 | 17778 | 3.27656 | 3.29930 | 0.17944 | Separate ban/pick heads |
| exp-league-team-features | 20260111_011015 | 20 | 142224 | 17778 | 17778 | 3.12629 | 3.15124 | 0.18776 | Feature set: draft_sequence, patch, action_type, side, event_index, league, team |
| exp-picks-only | 20260111_012429 | 20 | 71112 | 8889 | 8889 | 3.21665 | 3.29753 | 0.18045 | Picks-only subset |
