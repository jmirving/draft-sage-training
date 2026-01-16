from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from draft_sage_training.utils.champion_sanitizer import ChampionSanitizer

MISSING_TIMESTAMP = np.iinfo(np.int64).max


def _normalize_category(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def load_champion_eligibility(
    path: Optional[str],
    sanitizer: ChampionSanitizer,
    champion2idx: dict[str, int],
) -> dict[str, np.ndarray]:
    if not path:
        return {}

    eligibility_path = Path(path)
    if not eligibility_path.exists():
        raise FileNotFoundError(f"Champion eligibility artifact not found: {eligibility_path}")

    with eligibility_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    leagues_payload = payload.get("leagues")
    if not isinstance(leagues_payload, dict):
        raise ValueError("Champion eligibility artifact must include a leagues mapping.")

    by_league: dict[str, np.ndarray] = {}
    for league_name, league_payload in leagues_payload.items():
        if not isinstance(league_payload, dict):
            continue
        first_seen = league_payload.get("first_seen")
        if not isinstance(first_seen, dict):
            continue

        normalized_league = _normalize_category(league_name)
        if not normalized_league:
            continue

        vector = np.full(len(champion2idx) - 1, MISSING_TIMESTAMP, dtype=np.int64)
        for champ_name, date_value in first_seen.items():
            if champ_name is None or date_value is None:
                continue
            sanitized = sanitizer.sanitize(str(champ_name))
            if not sanitized:
                continue
            idx = champion2idx.get(sanitized)
            if idx is None or idx == 0:
                raise ValueError(f"Unknown champion in eligibility artifact: {champ_name}")
            parsed = pd.to_datetime(date_value, errors="coerce")
            if pd.isna(parsed):
                raise ValueError(
                    f"Invalid date for {champ_name} in eligibility artifact: {date_value}"
                )
            vector[idx - 1] = parsed.value

        by_league[normalized_league] = vector

    return by_league
