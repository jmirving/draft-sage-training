from __future__ import annotations

import argparse
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from draft_sage_training.config import TrainingConfig, normalize_patches
from draft_sage_training.dataset import DraftDataset
from draft_sage_training.model import DraftMLP
from draft_sage_training.training import (
    build_split_groups,
    evaluate_model,
    get_optimizer,
    run_epoch,
    set_seed,
    split_indices,
)
from draft_sage_training.utils.champion_mapping import DEFAULT_MAPPING_PATH
from draft_sage_training.utils.draft_order import DRAFT_ORDER


@dataclass(frozen=True)
class DiagnosticConfig:
    subset_size: int
    overfit_epochs: int
    shuffle_epochs: int
    skip_overfit: bool
    skip_shuffle: bool
    skip_leakage: bool
    skip_mask_check: bool


class ShuffledTargetDataset(Dataset):
    def __init__(self, base: Dataset, shuffled_targets: list[int]):
        self.base = base
        self.shuffled_targets = shuffled_targets

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = dict(self.base[idx])
        target = self.shuffled_targets[idx]
        sample["target"] = torch.tensor(target, dtype=sample["target"].dtype)
        return sample


def build_model(dataset: DraftDataset, device: torch.device) -> DraftMLP:
    model = DraftMLP(
        feature_dims={
            "num_champions": dataset.num_champions,
            "draft_sequence": dataset.draft_features,
            "num_patches": dataset.num_patches,
            "num_actions": 2,
            "num_sides": 2,
            "num_events": len(DRAFT_ORDER),
            "num_leagues": dataset.num_leagues,
            "num_teams": dataset.num_teams,
        },
        hidden_size=256,
        output_size=dataset.num_champions - 1,
    )
    return model.to(device)


def compute_uniform_chance(dataset: DraftDataset, indices: Iterable[int]) -> float:
    chances = []
    for idx in indices:
        sample = dataset.samples[idx]
        mask = dataset.get_output_mask(
            sample["already_picked_or_banned"],
            league_key=sample.get("league_key"),
            game_date_value=sample.get("game_date_value"),
        )
        available = float(np.sum(mask))
        if available > 0:
            chances.append(1.0 / available)
    return float(np.mean(chances)) if chances else 0.0


def compute_majority_baseline(dataset: DraftDataset, indices: Iterable[int]) -> tuple[float, Optional[str]]:
    targets = [dataset.samples[idx]["target"] for idx in indices]
    if not targets:
        return 0.0, None
    counter = Counter(targets)
    top_target, top_count = counter.most_common(1)[0]
    champ_name = dataset.idx2champion.get(top_target)
    return top_count / len(targets), champ_name


def check_target_masks(dataset: DraftDataset) -> int:
    violations = 0
    for sample in dataset.samples:
        target_index = sample["target"] - 1
        mask = dataset.get_output_mask(
            sample["already_picked_or_banned"],
            league_key=sample.get("league_key"),
            game_date_value=sample.get("game_date_value"),
        )
        if target_index < 0 or target_index >= len(mask) or mask[target_index] == 0:
            violations += 1
    return violations


def get_unique_keys(dataset: DraftDataset, indices: Iterable[int], key: str) -> set:
    values = set()
    for idx in indices:
        value = dataset.samples[idx].get(key)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            values.add(value)
    return values


def summarize_overlap(dataset: DraftDataset, train_indices: list[int], val_indices: list[int], test_indices: list[int]):
    summary = {}
    for key in ("gameid", "seriesid"):
        train_set = get_unique_keys(dataset, train_indices, key)
        val_set = get_unique_keys(dataset, val_indices, key)
        test_set = get_unique_keys(dataset, test_indices, key)
        summary[key] = {
            "train": len(train_set),
            "val": len(val_set),
            "test": len(test_set),
            "train_val_overlap": len(train_set & val_set),
            "train_test_overlap": len(train_set & test_set),
            "val_test_overlap": len(val_set & test_set),
        }
    return summary


def select_subset(indices: list[int], subset_size: int, seed: int) -> list[int]:
    if subset_size <= 0:
        return []
    if subset_size >= len(indices):
        return list(indices)
    rng = np.random.default_rng(seed)
    return rng.choice(indices, size=subset_size, replace=False).tolist()


def train_subset(
    dataset: DraftDataset,
    indices: list[int],
    config: TrainingConfig,
    epochs: int,
    device: torch.device,
    shuffle_targets: bool = False,
) -> tuple[float, float]:
    subset = Subset(dataset, indices)

    if shuffle_targets:
        targets = [int(subset[i]["target"].item()) for i in range(len(subset))]
        rng = np.random.default_rng(config.seed)
        rng.shuffle(targets)
        subset = ShuffledTargetDataset(subset, targets)

    batch_size = min(config.batch_size, len(subset)) or 1
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    model = build_model(dataset, device)
    optimizer = get_optimizer(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    for _ in range(max(epochs, 1)):
        run_epoch(model, loader, loss_function, optimizer, device=device, is_train=True)

    loss, accuracy = evaluate_model(model, eval_loader, loss_function, device=device)
    return loss, accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DraftSage training diagnostics (leakage, mask checks, overfit, shuffle).",
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
        help="Directory containing per-patch champion priors JSON files.",
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
        help="Directory containing per-patch role priors JSON files.",
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
        help="Number of training epochs (unused in diagnostics).",
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
        default="diagnostics",
        help="Category label for diagnostics.",
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
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Verbosity for log messages.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=256,
        help="Subset size for overfit/shuffle checks.",
    )
    parser.add_argument(
        "--overfit-epochs",
        type=int,
        default=40,
        help="Epochs for the overfit subset test.",
    )
    parser.add_argument(
        "--shuffle-epochs",
        type=int,
        default=20,
        help="Epochs for the shuffled-label subset test.",
    )
    parser.add_argument(
        "--skip-overfit",
        action="store_true",
        help="Skip the overfit subset test.",
    )
    parser.add_argument(
        "--skip-shuffle",
        action="store_true",
        help="Skip the shuffled-label subset test.",
    )
    parser.add_argument(
        "--skip-leakage",
        action="store_true",
        help="Skip train/val/test leakage checks.",
    )
    parser.add_argument(
        "--skip-mask-check",
        action="store_true",
        help="Skip output-mask integrity checks.",
    )
    return parser


def build_config(args: argparse.Namespace) -> TrainingConfig:
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
        use_league_team_embeddings=not args.no_league_team_embeddings,
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
    )


def build_diag_config(args: argparse.Namespace) -> DiagnosticConfig:
    return DiagnosticConfig(
        subset_size=args.subset_size,
        overfit_epochs=args.overfit_epochs,
        shuffle_epochs=args.shuffle_epochs,
        skip_overfit=args.skip_overfit,
        skip_shuffle=args.skip_shuffle,
        skip_leakage=args.skip_leakage,
        skip_mask_check=args.skip_mask_check,
    )


def run_diagnostics(config: TrainingConfig, diag_config: DiagnosticConfig) -> int:
    if config.patch_window and config.patches:
        raise ValueError("Provide either patch_window or patches, not both.")

    split_total = config.train_split + config.val_split + config.test_split
    if not np.isclose(split_total, 1.0, rtol=1e-4):
        raise ValueError("Train/val/test splits must sum to 1.0.")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    dataset = DraftDataset(
        input_dir=config.input_dir,
        patch_window=config.patch_window,
        patches=config.patches,
        champion_mapping_path=config.champion_mapping_path,
        champion_eligibility_path=config.champion_eligibility_path,
        champion_priors_dir=config.champion_priors_dir,
        champion_priors_strength=config.champion_priors_strength,
        champion_priors_time_buckets=config.champion_priors_time_buckets,
        role_priors_dir=config.role_priors_dir,
        role_priors_strength=config.role_priors_strength,
        use_league_team_embeddings=config.use_league_team_embeddings,
    )
    if len(dataset) == 0:
        logging.error("No training samples available.")
        return 1

    groups = build_split_groups(dataset, config.split_strategy)
    train_indices, val_indices, test_indices = split_indices(
        len(dataset),
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed,
        groups=groups,
    )

    logging.info("Total samples: %d", len(dataset))
    logging.info(
        "Split sizes: train=%d val=%d test=%d",
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

    uniform_chance = compute_uniform_chance(dataset, test_indices)
    majority_baseline, majority_champ = compute_majority_baseline(dataset, test_indices)
    logging.info("Uniform chance (mask-aware) on test: %.5f", uniform_chance)
    if majority_champ:
        logging.info(
            "Majority-class baseline on test: %.5f (champion=%s)",
            majority_baseline,
            majority_champ,
        )

    if not diag_config.skip_mask_check:
        violations = check_target_masks(dataset)
        if violations:
            logging.warning("Mask integrity violations: %d", violations)
        else:
            logging.info("Mask integrity: OK")

    if not diag_config.skip_leakage:
        overlap_summary = summarize_overlap(dataset, train_indices, val_indices, test_indices)
        for key, summary in overlap_summary.items():
            logging.info(
                "%s overlap: train=%d val=%d test=%d train/val=%d train/test=%d val/test=%d",
                key,
                summary["train"],
                summary["val"],
                summary["test"],
                summary["train_val_overlap"],
                summary["train_test_overlap"],
                summary["val_test_overlap"],
            )

    subset_indices = select_subset(train_indices, diag_config.subset_size, config.seed)
    if not subset_indices:
        logging.warning("Subset size is 0; skipping overfit/shuffle diagnostics.")
        return 0

    if not diag_config.skip_overfit:
        overfit_loss, overfit_accuracy = train_subset(
            dataset,
            subset_indices,
            config,
            diag_config.overfit_epochs,
            device,
            shuffle_targets=False,
        )
        logging.info(
            "Overfit subset: loss=%.4f accuracy=%.4f (epochs=%d size=%d)",
            overfit_loss,
            overfit_accuracy,
            diag_config.overfit_epochs,
            len(subset_indices),
        )

    if not diag_config.skip_shuffle:
        shuffle_loss, shuffle_accuracy = train_subset(
            dataset,
            subset_indices,
            config,
            diag_config.shuffle_epochs,
            device,
            shuffle_targets=True,
        )
        logging.info(
            "Shuffle subset: loss=%.4f accuracy=%.4f (epochs=%d size=%d)",
            shuffle_loss,
            shuffle_accuracy,
            diag_config.shuffle_epochs,
            len(subset_indices),
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args)
    diag_config = build_diag_config(args)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return run_diagnostics(config, diag_config)


if __name__ == "__main__":
    raise SystemExit(main())
