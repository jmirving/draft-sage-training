#!/usr/bin/env python3
"""Generate per-league champion eligibility cutoffs from processed team data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from draft_sage_training.dataset import BAN_COLUMNS, PICK_COLUMNS, TEAM_IDS, load_processed_teams
from draft_sage_training.utils.champion_mapping import load_champion_mapping
from draft_sage_training.utils.champion_sanitizer import ChampionSanitizer

DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / ".tmp" / "prodata-2025-plus-2026-01-clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-league champion eligibility dates from picks/bans.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=(
            "Processed prodata directory (or direct teams CSV folder). "
            "Defaults to cleaned 2025 + Jan 2026 dataset."
        ),
    )
    parser.add_argument(
        "--output-path",
        default="data/eligibility/champion-eligibility.json",
        help="Output path for the eligibility artifact JSON.",
    )
    parser.add_argument(
        "--champion-mapping-path",
        required=True,
        help="Path to champion-mapping latest.json artifact.",
    )
    parser.add_argument(
        "--no-picks",
        action="store_true",
        help="Exclude picks from the eligibility calculation.",
    )
    parser.add_argument(
        "--no-bans",
        action="store_true",
        help="Exclude bans from the eligibility calculation.",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional note to include in the artifact metadata.",
    )
    return parser.parse_args()


def normalize_category(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def build_champion_lookup(mapping_entries, sanitizer: ChampionSanitizer) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for entry in mapping_entries:
        normalized = entry.get("normalized_name")
        if not normalized:
            continue
        sanitized = sanitizer.sanitize(str(normalized))
        if not sanitized:
            continue
        lookup[sanitized] = str(normalized)
    return lookup


def main() -> None:
    args = parse_args()
    include_picks = not args.no_picks
    include_bans = not args.no_bans

    if not include_picks and not include_bans:
        raise ValueError("At least one of picks or bans must be included.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    teams_df = load_processed_teams(args.input_dir)
    if "participantid" in teams_df.columns:
        teams_df = teams_df[teams_df["participantid"].isin(list(TEAM_IDS))].copy()

    if "league" not in teams_df.columns or "date" not in teams_df.columns:
        raise ValueError("Teams data must include league and date columns.")

    pick_columns = [col for col in PICK_COLUMNS if col in teams_df.columns] if include_picks else []
    ban_columns = [col for col in BAN_COLUMNS if col in teams_df.columns] if include_bans else []
    if not pick_columns and not ban_columns:
        raise ValueError("No pick/ban columns available in teams data.")

    sanitizer = ChampionSanitizer()
    mapping_entries = load_champion_mapping(args.champion_mapping_path)
    champion_lookup = build_champion_lookup(mapping_entries, sanitizer)

    first_seen_by_league: dict[str, dict[str, pd.Timestamp]] = {}
    unknown_champions = Counter()
    missing_dates = 0
    missing_leagues = 0

    for _, row in teams_df.iterrows():
        league = normalize_category(row.get("league"))
        if not league:
            missing_leagues += 1
            continue

        parsed_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(parsed_date):
            missing_dates += 1
            continue

        league_entries = first_seen_by_league.setdefault(league, {})
        for column in pick_columns + ban_columns:
            champion_name = row.get(column)
            if champion_name is None or pd.isna(champion_name):
                continue
            sanitized = sanitizer.sanitize(champion_name)
            if not sanitized:
                continue
            normalized = champion_lookup.get(sanitized)
            if not normalized:
                unknown_champions[sanitized] += 1
                continue

            existing = league_entries.get(normalized)
            if existing is None or parsed_date < existing:
                league_entries[normalized] = parsed_date

    if missing_leagues:
        logging.info("Skipped %d rows missing league data.", missing_leagues)
    if missing_dates:
        logging.info("Skipped %d rows with invalid dates.", missing_dates)
    if unknown_champions:
        logging.info(
            "Unknown champions encountered: %d entries (%d unique).",
            sum(unknown_champions.values()),
            len(unknown_champions),
        )

    leagues_payload: dict[str, dict[str, dict[str, str]]] = {}
    for league in sorted(first_seen_by_league.keys()):
        entries = first_seen_by_league[league]
        first_seen = {
            champion: entries[champion].date().isoformat() for champion in sorted(entries.keys())
        }
        leagues_payload[league] = {"first_seen": first_seen}

    payload = {
        "schema_version": "1.0",
        "eligibility_type": "champion",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "leagues": leagues_payload,
        "meta": {
            "input_dir": args.input_dir,
            "include_picks": include_picks,
            "include_bans": include_bans,
            "note": args.note,
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logging.info("Wrote champion eligibility artifact to %s", output_path)


if __name__ == "__main__":
    main()
