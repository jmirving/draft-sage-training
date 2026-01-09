"""Utilities for normalising champion names across different data sources."""

from __future__ import annotations

import unicodedata
from typing import Optional


class ChampionSanitizer:
    """Apply consistent transformations to champion names used throughout training."""

    def sanitize(self, champion_name: Optional[str]) -> str:
        if champion_name is None:
            return ""

        cleaned = str(champion_name).strip()
        if not cleaned:
            return ""

        cleaned = cleaned.upper()
        cleaned = "".join(ch for ch in cleaned if not self._is_punctuation(ch))
        cleaned = " ".join(cleaned.split())
        return cleaned

    @staticmethod
    def _is_punctuation(character: str) -> bool:
        return unicodedata.category(character).startswith("P")
