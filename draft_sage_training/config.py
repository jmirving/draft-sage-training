from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class TrainingConfig:
    input_dir: str
    output_dir: str
    train_split: float
    val_split: float
    test_split: float
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    patch_window: Optional[int]
    patches: Optional[Sequence[str]]
    log_level: str


def normalize_patches(patches: Optional[Iterable[str]]) -> Optional[Sequence[str]]:
    if not patches:
        return None
    normalized = [str(patch).strip() for patch in patches if patch]
    return normalized or None
