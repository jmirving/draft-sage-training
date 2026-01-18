from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Subset

from draft_sage_training.config import TrainingConfig
from draft_sage_training.dataset import DraftDataset
from draft_sage_training.model import DraftMLP
from draft_sage_training.utils.draft_order import DRAFT_ORDER


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_optimizer(parameters, lr: float):
    import torch.optim as optim

    return optim.Adam(parameters, lr=lr)


def run_epoch(model, data_loader, loss_function, optimizer=None, device="cpu", is_train=True) -> float:
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for batch in data_loader:
            features = {
                "draft_sequence": batch["draft_sequence"].to(device),
                "patch_index": batch["patch_index"].to(device),
                "action_type": batch["action_type"].to(device),
                "side": batch["side"].to(device),
                "event_index": batch["event_index"].to(device),
                "league_index": batch["league_index"].to(device),
                "team_index": batch["team_index"].to(device),
            }
            if "champion_priors" in batch:
                features["champion_priors"] = batch["champion_priors"].to(device)
            if "role_priors" in batch:
                features["role_priors"] = batch["role_priors"].to(device)
            outputs = model(features)
            masked_outputs = outputs.masked_fill(batch["output_mask"].to(device) == 0, -1e9)
            loss = loss_function(masked_outputs, batch["target"].to(device))
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(data_loader) if len(data_loader) else 0.0


def evaluate_model(model, data_loader, loss_function, device="cpu") -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            features = {
                "draft_sequence": batch["draft_sequence"].to(device),
                "patch_index": batch["patch_index"].to(device),
                "action_type": batch["action_type"].to(device),
                "side": batch["side"].to(device),
                "event_index": batch["event_index"].to(device),
                "league_index": batch["league_index"].to(device),
                "team_index": batch["team_index"].to(device),
            }
            if "champion_priors" in batch:
                features["champion_priors"] = batch["champion_priors"].to(device)
            if "role_priors" in batch:
                features["role_priors"] = batch["role_priors"].to(device)
            outputs = model(features)
            masked_outputs = outputs.masked_fill(batch["output_mask"].to(device) == 0, -1e9)
            loss = loss_function(masked_outputs, batch["target"].to(device))
            total_loss += loss.item()

            predictions = masked_outputs.argmax(dim=1)
            correct_predictions += (predictions == batch["target"].to(device)).sum().item()
            total_examples += batch["target"].size(0)

    avg_loss = total_loss / len(data_loader) if len(data_loader) else 0.0
    accuracy = (correct_predictions / total_examples) if total_examples else 0.0
    return avg_loss, accuracy


def split_indices(
    total_size: int,
    train_split: float,
    val_split: float,
    test_split: float,
    seed: int,
    groups: Optional[Sequence[str]] = None,
):
    indices = np.arange(total_size)
    if not groups:
        train_val, test = train_test_split(
            indices,
            test_size=test_split,
            random_state=seed,
            shuffle=True,
        )
        val_relative = val_split / (train_split + val_split)
        train, val = train_test_split(
            train_val,
            test_size=val_relative,
            random_state=seed,
            shuffle=True,
        )
        return train, val, test

    if len(groups) != total_size:
        raise ValueError("Group labels must align with dataset size.")

    groups_array = np.asarray(groups)
    train_val_split = GroupShuffleSplit(
        n_splits=1,
        test_size=test_split,
        random_state=seed,
    )
    train_val_idx, test_idx = next(train_val_split.split(indices, groups=groups_array))

    val_relative = val_split / (train_split + val_split)
    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_relative,
        random_state=seed,
    )
    train_idx, val_idx = next(
        val_splitter.split(train_val_idx, groups=groups_array[train_val_idx])
    )
    train = indices[train_val_idx][train_idx]
    val = indices[train_val_idx][val_idx]
    test = indices[test_idx]
    return train, val, test


def build_split_groups(dataset: DraftDataset, strategy: str) -> Optional[list[str]]:
    if not strategy or strategy == "random":
        return None
    if strategy not in {"gameid", "seriesid"}:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    groups = []
    for index, sample in enumerate(dataset.samples):
        value = sample.get(strategy)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            groups.append(f"missing-{index}")
        else:
            groups.append(str(value))
    return groups


def run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_run_id_timestamp(run_identifier: str) -> str | None:
    try:
        parsed = datetime.strptime(run_identifier, "%Y%m%d_%H%M%S")
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def build_dataset_meta(config: TrainingConfig) -> dict:
    label = config.dataset_label
    if not label:
        label = Path(config.input_dir).name or config.input_dir

    dataset = {"label": label, "input_dir": config.input_dir}
    if config.champion_eligibility_path:
        dataset["champion_eligibility"] = {"path": config.champion_eligibility_path}
    if config.champion_priors_dir:
        dataset["champion_priors"] = {
            "dir": config.champion_priors_dir,
            "strength": config.champion_priors_strength,
            "time_buckets": config.champion_priors_time_buckets,
        }
    if config.role_priors_dir:
        dataset["role_priors"] = {
            "dir": config.role_priors_dir,
            "strength": config.role_priors_strength,
        }
    if config.early_blue_ban_priors:
        dataset["early_blue_ban_priors"] = {
            "strength": config.early_blue_ban_priors_strength,
            "window_days": config.team_priors_window_days,
        }
    if config.patch_window:
        dataset["patch_window"] = config.patch_window
    if config.patches:
        dataset["patches"] = list(config.patches)
    return dataset


def build_summary_payload(
    *,
    run_identifier: str,
    status: str,
    created_at: str,
    updated_at: str,
    description: str | None,
    category: str,
    display_name: str,
    dataset_meta: dict,
    epochs: int,
    metrics: dict | None,
    samples: dict | None,
    paths: dict | None,
    progress_epoch: int | None = None,
) -> dict:
    progress = None
    if progress_epoch is not None:
        progress = {"epoch": progress_epoch, "epochs": epochs}
    return {
        "schema_version": "1.0",
        "run_id": run_identifier,
        "display_name": display_name,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "description": description,
        "category": category,
        "dataset": dataset_meta,
        "progress": progress,
        "metrics": metrics,
        "samples": samples,
        "paths": paths,
    }


def update_experiment_index(index_path: Path, run_entry: dict) -> None:
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
    else:
        index_data = {"schema_version": "1.0", "generated_at": None, "runs": []}

    runs = index_data.get("runs")
    if not isinstance(runs, list):
        runs = []

    run_id_value = run_entry.get("run_id")
    if not run_id_value:
        raise ValueError("summary.json must include run_id")

    runs = [entry for entry in runs if entry.get("run_id") != run_id_value]
    runs.append(run_entry)

    index_data["schema_version"] = index_data.get("schema_version") or "1.0"
    index_data["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    index_data["runs"] = sorted(runs, key=lambda entry: entry.get("run_id") or "")

    write_json(index_path, index_data)


def train(config: TrainingConfig) -> int:
    if config.patch_window and config.patches:
        raise ValueError("Provide either patch_window or patches, not both.")

    split_total = config.train_split + config.val_split + config.test_split
    if not math.isclose(split_total, 1.0, rel_tol=1e-4):
        raise ValueError("Train/val/test splits must sum to 1.0.")

    run_identifier = run_id()
    created_at = parse_run_id_timestamp(run_identifier)
    if not created_at:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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
        early_blue_ban_priors=config.early_blue_ban_priors,
        early_blue_ban_priors_strength=config.early_blue_ban_priors_strength,
        team_priors_window_days=config.team_priors_window_days,
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
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    run_dir = Path(config.output_dir) / run_identifier
    model_path = run_dir / "model.pth"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_path, asdict(config))

    dataset_meta = build_dataset_meta(config)
    base_paths = {
        "config": config_path.name,
        "metrics": metrics_path.name,
        "model": model_path.name,
    }
    running_summary = build_summary_payload(
        run_identifier=run_identifier,
        status="running",
        created_at=created_at,
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=config.description,
        category=config.category,
        display_name=config.display_name or run_identifier,
        dataset_meta=dataset_meta,
        epochs=config.epochs,
        metrics={"accuracy": None, "loss": None, "best_val_loss": None},
        samples={
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        paths=base_paths,
        progress_epoch=0,
    )
    write_json(summary_path, running_summary)
    if config.update_index:
        index_path = Path(config.output_dir) / "experiment-index.json"
        run_entry = {
            "run_id": run_identifier,
            "display_name": running_summary["display_name"],
            "status": running_summary["status"],
            "category": running_summary["category"],
            "dataset": dataset_meta,
            "metrics": {"accuracy": None, "loss": None},
            "summary_path": str(summary_path.relative_to(Path(config.output_dir))),
        }
        update_experiment_index(index_path, run_entry)
        logging.info("Marked run as running in %s", index_path)

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
    ).to(device)
    optimizer = get_optimizer(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_model_state = None

    logging.info("Starting training for %s epochs.", config.epochs)
    for epoch in range(config.epochs):
        train_loss = run_epoch(model, train_loader, loss_function, optimizer, device=device, is_train=True)
        val_loss = run_epoch(model, val_loader, loss_function, optimizer=None, device=device, is_train=False)
        logging.info("Epoch %s: train_loss=%.4f val_loss=%.4f", epoch + 1, train_loss, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        running_summary["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        running_summary["progress"] = {"epoch": epoch + 1, "epochs": config.epochs}
        write_json(summary_path, running_summary)

    if best_model_state is None:
        logging.error("Training did not produce a model state.")
        return 1

    model.load_state_dict(best_model_state)
    test_loss, test_accuracy = evaluate_model(model, test_loader, loss_function, device=device)
    torch.save(best_model_state, model_path)

    feature_set = [
        "draft_sequence",
        "patch",
        "action_type",
        "side",
        "event_index",
    ]
    if config.use_league_team_embeddings:
        feature_set.extend(["league", "team"])
    if config.champion_priors_dir:
        feature_set.append("champion_priors")
    if config.role_priors_dir:
        feature_set.append("role_priors")
    if config.early_blue_ban_priors:
        feature_set.append("early_blue_ban_priors")
    metrics_payload = {
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "feature_set": feature_set,
        "num_leagues": dataset.num_leagues,
        "num_teams": dataset.num_teams,
        "champion_priors_time_buckets": config.champion_priors_time_buckets,
    }
    if config.role_priors_dir:
        metrics_payload["role_priors_strength"] = config.role_priors_strength
    if config.early_blue_ban_priors:
        metrics_payload["early_blue_ban_priors_strength"] = config.early_blue_ban_priors_strength
        metrics_payload["team_priors_window_days"] = config.team_priors_window_days
    write_json(metrics_path, metrics_payload)

    summary_payload = build_summary_payload(
        run_identifier=run_identifier,
        status="completed",
        created_at=created_at,
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        description=config.description,
        category=config.category,
        display_name=config.display_name or run_identifier,
        dataset_meta=dataset_meta,
        epochs=config.epochs,
        metrics={
            "accuracy": test_accuracy,
            "loss": test_loss,
            "best_val_loss": best_val_loss,
        },
        samples={
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        paths=base_paths,
        progress_epoch=config.epochs,
    )
    write_json(summary_path, summary_payload)

    logging.info("Saved model to %s", model_path)
    logging.info("Saved metrics to %s", metrics_path)
    logging.info("Saved summary to %s", summary_path)

    if config.update_index:
        index_path = Path(config.output_dir) / "experiment-index.json"
        run_entry = {
            "run_id": run_identifier,
            "display_name": summary_payload["display_name"],
            "status": summary_payload["status"],
            "category": summary_payload["category"],
            "dataset": dataset_meta,
            "metrics": {
                "accuracy": test_accuracy,
                "loss": test_loss,
            },
            "summary_path": str(summary_path.relative_to(Path(config.output_dir))),
        }
        update_experiment_index(index_path, run_entry)
        logging.info("Updated experiment index at %s", index_path)
    else:
        logging.info("Skipping experiment index update.")
    return 0
