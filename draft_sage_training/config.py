from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class TrainingConfig:
    input_dir: str
    output_dir: str
    champion_mapping_path: str
    champion_eligibility_path: Optional[str]
    champion_priors_dir: Optional[str]
    champion_priors_strength: float
    champion_priors_time_buckets: int
    role_priors_dir: Optional[str]
    role_priors_strength: float
    use_league_embeddings: bool
    use_team_embeddings: bool
    train_split: float
    val_split: float
    test_split: float
    split_strategy: str
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    patch_window: Optional[int]
    patches: Optional[Sequence[str]]
    category: str
    display_name: Optional[str]
    description: Optional[str]
    dataset_label: Optional[str]
    update_index: bool
    log_level: str
    publish_data_dir: Optional[str]
    publish_indexes: Optional[Sequence[str]]
    publish_on_start: bool
    publish_on_finish: bool
    publish_commit: bool
    publish_push: bool
    inspection_keep: int


def normalize_patches(patches: Optional[Iterable[str]]) -> Optional[Sequence[str]]:
    if not patches:
        return None
    normalized = [str(patch).strip() for patch in patches if patch]
    return normalized or None
