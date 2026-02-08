from __future__ import annotations

import argparse
import logging

from draft_sage_training.config import TrainingConfig, normalize_patches
from draft_sage_training.utils.champion_mapping import DEFAULT_MAPPING_PATH
from draft_sage_training.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train DraftSage models using processed Oracle's Elixir data.",
    )
    parser.add_argument(
        "--input-dir",
        default="data/prodata",
        help="Directory containing lol-pro-data-processor outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for training outputs (models/metrics/config).",
    )
    parser.add_argument(
        "--champion-mapping-path",
        default=DEFAULT_MAPPING_PATH,
        help="Path to the DDragon champion mapping artifact JSON.",
    )
    parser.add_argument(
        "--champion-eligibility-path",
        help="Optional path to the per-league champion eligibility artifact JSON.",
    )
    parser.add_argument(
        "--champion-priors-dir",
        help="Directory containing per-patch pick/ban (champion) priors JSON files.",
    )
    parser.add_argument(
        "--champion-priors-strength",
        type=float,
        default=1.0,
        help="Scale factor applied to champion priors when used as logit bias.",
    )
    parser.add_argument(
        "--champion-priors-time-buckets",
        type=int,
        default=1,
        help="Time buckets per patch for champion priors (1 disables time-aware priors).",
    )
    parser.add_argument(
        "--role-priors-dir",
        help="Directory containing per-patch champion role-distribution priors JSON files.",
    )
    parser.add_argument(
        "--role-priors-strength",
        type=float,
        default=1.0,
        help="Scale factor applied to role priors when used as logit bias.",
    )
    parser.add_argument(
        "--no-league-team-embeddings",
        action="store_true",
        help="Disable league/team embeddings (use unknown indices only).",
    )
    parser.add_argument(
        "--no-league-embeddings",
        action="store_true",
        help="Disable league embeddings (use unknown league index only).",
    )
    parser.add_argument(
        "--no-team-embeddings",
        action="store_true",
        help="Disable team embeddings (use unknown team index only).",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data used for training.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data used for validation.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="Fraction of data used for testing.",
    )
    parser.add_argument(
        "--split-strategy",
        default="seriesid",
        choices=["random", "gameid", "seriesid"],
        help="How to split train/val/test (random or grouped by id).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for the data loader.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--patch-window",
        type=int,
        help="Number of most-recent patches to include.",
    )
    parser.add_argument(
        "--patch",
        dest="patches",
        action="append",
        help="Explicit patch to include; repeatable.",
    )
    parser.add_argument(
        "--category",
        default="uncategorized",
        help="Experiment category for UI grouping.",
    )
    parser.add_argument(
        "--display-name",
        help="Friendly display name for the run.",
    )
    parser.add_argument(
        "--description",
        help="Short description for the run summary.",
    )
    parser.add_argument(
        "--dataset-label",
        help="Friendly dataset label (e.g., 'Clean 2025').",
    )
    parser.add_argument(
        "--no-index-update",
        action="store_true",
        help="Skip updating experiment-index.json.",
    )
    parser.add_argument(
        "--publish-data",
        action="store_true",
        help="Publish run metadata to the data host on start and finish (commit + push).",
    )
    parser.add_argument(
        "--publish-on-start",
        action="store_true",
        help="Publish a running marker at the start of training.",
    )
    parser.add_argument(
        "--publish-on-finish",
        action="store_true",
        help="Publish completed artifacts after training finishes.",
    )
    parser.add_argument(
        "--publish-data-dir",
        help="Target data host training directory.",
    )
    parser.add_argument(
        "--publish-index",
        action="append",
        help="Experiment index path(s) to publish (repeatable). Defaults to output-dir index.",
    )
    parser.add_argument(
        "--publish-commit",
        action="store_true",
        help="Commit data host changes after publishing.",
    )
    parser.add_argument(
        "--publish-push",
        action="store_true",
        help="Push data host changes after committing.",
    )
    parser.add_argument(
        "--inspection-keep",
        type=int,
        default=10,
        help="Number of newest inspection bundles to keep on the data host.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Verbosity for log messages.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    publish_on_start = args.publish_on_start or args.publish_data
    publish_on_finish = args.publish_on_finish or args.publish_data
    publish_commit = args.publish_commit or args.publish_data
    publish_push = args.publish_push or args.publish_data
    if publish_push and not publish_commit:
        publish_commit = True
    use_league_embeddings = not args.no_league_embeddings
    use_team_embeddings = not args.no_team_embeddings
    if args.no_league_team_embeddings:
        use_league_embeddings = False
        use_team_embeddings = False
    return TrainingConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        champion_mapping_path=args.champion_mapping_path,
        champion_eligibility_path=args.champion_eligibility_path,
        champion_priors_dir=args.champion_priors_dir,
        champion_priors_strength=args.champion_priors_strength,
        champion_priors_time_buckets=args.champion_priors_time_buckets,
        role_priors_dir=args.role_priors_dir,
        role_priors_strength=args.role_priors_strength,
        use_league_embeddings=use_league_embeddings,
        use_team_embeddings=use_team_embeddings,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        split_strategy=args.split_strategy,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patch_window=args.patch_window,
        patches=normalize_patches(args.patches),
        category=args.category,
        display_name=args.display_name,
        description=args.description,
        dataset_label=args.dataset_label,
        update_index=not args.no_index_update,
        log_level=args.log_level,
        publish_data_dir=args.publish_data_dir,
        publish_indexes=args.publish_index,
        publish_on_start=publish_on_start,
        publish_on_finish=publish_on_finish,
        publish_commit=publish_commit,
        publish_push=publish_push,
        inspection_keep=args.inspection_keep,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return train(config)


if __name__ == "__main__":
    raise SystemExit(main())
