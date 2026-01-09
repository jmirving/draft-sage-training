from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from draft_sage_training.config import TrainingConfig
from draft_sage_training.dataset import DraftDataset
from draft_sage_training.model import DraftMLP


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
            }
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
            }
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


def split_indices(total_size: int, train_split: float, val_split: float, test_split: float, seed: int):
    indices = np.arange(total_size)
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


def run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def train(config: TrainingConfig) -> int:
    if config.patch_window and config.patches:
        raise ValueError("Provide either patch_window or patches, not both.")

    split_total = config.train_split + config.val_split + config.test_split
    if not math.isclose(split_total, 1.0, rel_tol=1e-4):
        raise ValueError("Train/val/test splits must sum to 1.0.")

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    dataset = DraftDataset(
        input_dir=config.input_dir,
        patch_window=config.patch_window,
        patches=config.patches,
        champion_mapping_path=config.champion_mapping_path,
    )
    if len(dataset) == 0:
        logging.error("No training samples available.")
        return 1

    train_indices, val_indices, test_indices = split_indices(
        len(dataset),
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        seed=config.seed,
    )
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    model = DraftMLP(
        feature_dims={
            "num_champions": dataset.num_champions,
            "draft_sequence": dataset.draft_features,
            "num_patches": dataset.num_patches,
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

    if best_model_state is None:
        logging.error("Training did not produce a model state.")
        return 1

    model.load_state_dict(best_model_state)
    test_loss, test_accuracy = evaluate_model(model, test_loader, loss_function, device=device)

    run_dir = Path(config.output_dir) / run_id()
    model_path = run_dir / "model.pth"
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, model_path)
    write_json(config_path, asdict(config))
    write_json(
        metrics_path,
        {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "best_val_loss": best_val_loss,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
    )

    logging.info("Saved model to %s", model_path)
    logging.info("Saved metrics to %s", metrics_path)
    return 0
