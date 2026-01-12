# Training Accuracy Improvements

This document tracks experiment branches for improving training accuracy. Each experiment lives on its own branch and was trained with the same clean-data filters (complete picks, complete bans, blue/red rows).

## Baseline (main)
- Data: clean (all picks/bans, both sides)
- Command:
  `python scripts/train.py --input-dir /home/jirving/projects/lol/.tmp/training-real/prodata-processed --output-dir /home/jirving/projects/lol/.tmp/training-clean/artifacts --champion-mapping-path /home/jirving/projects/lol/.tmp/training-real/ddragon/artifacts/champion-mapping/latest.json --epochs 10 --batch-size 64`
- Metrics:
  - test_accuracy: 0.1742
  - test_loss: 3.2005
  - best_val_loss: 3.1977
  - samples: train 156955, val 19620, test 19620
- Artifacts: `/home/jirving/projects/lol/.tmp/training-clean/artifacts/20260110_015053/`

## Experiment 1: action/side/event embeddings
- Branch: `exp-action-side-event`
- Change: add embeddings for action type (ban/pick), side (blue/red), and event index.
- Command:
  `python scripts/train.py --input-dir /home/jirving/projects/lol/.tmp/training-real/prodata-processed --output-dir /home/jirving/projects/lol/.tmp/training-exp-action-side-event/artifacts --champion-mapping-path /home/jirving/projects/lol/.tmp/training-real/ddragon/artifacts/champion-mapping/latest.json --epochs 10 --batch-size 64`
- Metrics:
  - test_accuracy: 0.1821
  - test_loss: 3.1627
  - best_val_loss: 3.1654
  - samples: train 156955, val 19620, test 19620
- Artifacts: `/home/jirving/projects/lol/.tmp/training-exp-action-side-event/artifacts/20260110_021648/`

## Experiment 2: split ban/pick heads
- Branch: `exp-ban-pick-heads`
- Change: single encoder with two output heads; ban/pick head selected per sample.
- Command:
  `python scripts/train.py --input-dir /home/jirving/projects/lol/.tmp/training-real/prodata-processed --output-dir /home/jirving/projects/lol/.tmp/training-exp-ban-pick-heads/artifacts --champion-mapping-path /home/jirving/projects/lol/.tmp/training-real/ddragon/artifacts/champion-mapping/latest.json --epochs 10 --batch-size 64`
- Metrics:
  - test_accuracy: 0.1724
  - test_loss: 3.2617
  - best_val_loss: 3.2635
  - samples: train 156955, val 19620, test 19620
- Artifacts: `/home/jirving/projects/lol/.tmp/training-exp-ban-pick-heads/artifacts/20260110_022605/`

## Experiment 3: picks-only training
- Branch: `exp-picks-only`
- Change: emit only pick samples (bans still populate sequence context).
- Command:
  `python scripts/train.py --input-dir /home/jirving/projects/lol/.tmp/training-real/prodata-processed --output-dir /home/jirving/projects/lol/.tmp/training-exp-picks-only/artifacts --champion-mapping-path /home/jirving/projects/lol/.tmp/training-real/ddragon/artifacts/champion-mapping/latest.json --epochs 10 --batch-size 64`
- Metrics:
  - test_accuracy: 0.1931
  - test_loss: 3.1677
  - best_val_loss: 3.1462
  - samples: train 78480, val 9810, test 9810
- Artifacts: `/home/jirving/projects/lol/.tmp/training-exp-picks-only/artifacts/20260110_023541/`

## Parallel Run Plan
Run each experiment in a separate terminal to keep logs isolated:
1) `git_local checkout exp-action-side-event` then run the command above.
2) `git_local checkout exp-ban-pick-heads` then run the command above.
3) `git_local checkout exp-picks-only` then run the command above.

All three branches are pushed to origin and can be compared without modifying `main`.

## Clean 2025 priors sweep (exp-clean-2025-ep20)
- Data: clean 2025 (all picks/bans, both sides, complete blue/red rows)
- Features: action/side/event + league/team embeddings + champion priors
- Static priors best:
  - strength: 0.25
  - test_accuracy: 0.1924
  - test_loss: 3.1415
  - best_val_loss: 3.1164
  - artifacts: `/home/jirving/projects/lol/.tmp/training-clean-2025-league-team-priors-sweep/20260112_003737/`
- Static priors ranges tested: 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0 (best at 0.25).
- Time-aware priors (2 buckets) best:
  - strength: 0.5
  - test_accuracy: 0.1919
  - test_loss: 3.1515
  - best_val_loss: 3.1218
  - priors: `/home/jirving/projects/lol/.tmp/weights/champion-priors-timeaware-2/`
  - artifacts: `/home/jirving/projects/lol/.tmp/training-clean-2025-timeaware-priors-sweep/20260112_033410/`
- Additional time-aware (2 buckets) tested: strengths 0.15, 0.25, 1.0 (best still 0.5).
- No league/team embeddings (priors only) tested: strengths 0.5 and 4.0 (accuracies 0.1793–0.1798; worse than league/team).
- Decision: time-aware priors are a possible minor improvement, but current results do not beat the best static priors.
- UI combined index: `/home/jirving/projects/lol/.tmp/training-clean-2025-all/experiment-index.json`
