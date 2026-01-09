from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_MAPPING_PATH = "data/ddragon/artifacts/champion-mapping/latest.json"


def load_champion_mapping(path: str | None = None) -> list[dict[str, Any]]:
    mapping_path = Path(path or DEFAULT_MAPPING_PATH)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Champion mapping not found: {mapping_path}")
    with mapping_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Champion mapping payload must be a JSON array.")
    return data
