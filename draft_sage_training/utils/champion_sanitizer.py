"""Utilities for normalising champion names across different data sources."""

from __future__ import annotations

import unicodedata
from typing import Optional


class ChampionSanitizer:
    """Apply the DDragon name normalization rules used by champion mapping."""

    def sanitize(self, champion_name: Optional[str]) -> str:
        if champion_name is None:
            return ""

        cleaned = str(champion_name).strip()
        if not cleaned:
            return ""

        normalized = unicodedata.normalize("NFKD", cleaned)
        stripped = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
        lowered = stripped.lower()
        return "".join(ch for ch in lowered if ch.isalnum())
