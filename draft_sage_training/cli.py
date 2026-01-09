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
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Verbosity for log messages.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> TrainingConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    return TrainingConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        champion_mapping_path=args.champion_mapping_path,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patch_window=args.patch_window,
        patches=normalize_patches(args.patches),
        log_level=args.log_level,
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
