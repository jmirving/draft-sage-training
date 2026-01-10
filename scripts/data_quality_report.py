from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from draft_sage_training.dataset import BAN_COLUMNS, PICK_COLUMNS, TEAM_IDS, load_latest_csv
from draft_sage_training.utils.champion_mapping import DEFAULT_MAPPING_PATH, load_champion_mapping
from draft_sage_training.utils.champion_sanitizer import ChampionSanitizer


def is_missing(value: Any) -> bool:
    return value is None or pd.isna(value) or (isinstance(value, str) and value.strip() == "")


def counter_payload(counter: Counter, limit: int | None = None) -> list[dict[str, Any]]:
    items = counter.most_common(limit) if limit is not None else counter.most_common()
    return [{"key": str(key), "count": count} for key, count in items]


def load_known_champions(mapping_path: str, sanitizer: ChampionSanitizer) -> set[str] | None:
    try:
        mapping_entries = load_champion_mapping(mapping_path)
    except FileNotFoundError:
        return None

    known = set()
    for entry in mapping_entries:
        name = entry.get("normalized_name")
        if name:
            known.add(sanitizer.sanitize(name))
    return known


def build_report(input_dir: str, mapping_path: str, top_n: int) -> dict[str, Any]:
    input_path = Path(input_dir)
    teams_dir = input_path / "teams" if (input_path / "teams").is_dir() else input_path
    csv_path = load_latest_csv(teams_dir, ["teams_*.csv", "*.csv"])
    teams_df = pd.read_csv(csv_path)

    team_df = teams_df
    if "participantid" in teams_df.columns:
        team_mask = teams_df["participantid"].isin(list(TEAM_IDS))
        if team_mask.any():
            team_df = teams_df.loc[team_mask].copy()

    sanitizer = ChampionSanitizer()
    known_champions = load_known_champions(mapping_path, sanitizer)

    missing_pick_by_col = Counter()
    missing_pick_by_league = Counter()
    missing_pick_by_side = Counter()
    missing_pick_by_game = Counter()

    missing_ban_by_col = Counter()
    missing_ban_by_league = Counter()
    missing_ban_by_side = Counter()
    missing_ban_by_game = Counter()
    all_bans_missing_games = Counter()

    unknown_pick_names = Counter()
    unknown_ban_names = Counter()

    duplicate_picks = Counter()
    duplicate_bans = Counter()
    pick_ban_overlap = Counter()

    for gameid, rows in team_df.groupby("gameid"):
        picks = []
        bans = []
        for _, row in rows.iterrows():
            league = row.get("league") or "UNKNOWN"
            side = str(row.get("side") or "UNKNOWN").lower()

            for col in PICK_COLUMNS:
                value = row.get(col)
                if is_missing(value):
                    missing_pick_by_col[col] += 1
                    missing_pick_by_league[league] += 1
                    missing_pick_by_side[side] += 1
                    missing_pick_by_game[gameid] += 1
                else:
                    sanitized = sanitizer.sanitize(value)
                    if sanitized:
                        picks.append(sanitized)
                        if known_champions is not None and sanitized not in known_champions:
                            unknown_pick_names[str(value)] += 1

            missing_bans_for_row = 0
            for col in BAN_COLUMNS:
                value = row.get(col)
                if is_missing(value):
                    missing_ban_by_col[col] += 1
                    missing_ban_by_league[league] += 1
                    missing_ban_by_side[side] += 1
                    missing_ban_by_game[gameid] += 1
                    missing_bans_for_row += 1
                else:
                    sanitized = sanitizer.sanitize(value)
                    if sanitized:
                        bans.append(sanitized)
                        if known_champions is not None and sanitized not in known_champions:
                            unknown_ban_names[str(value)] += 1

            if missing_bans_for_row == len(BAN_COLUMNS):
                all_bans_missing_games[gameid] += 1

        if len(picks) != len(set(picks)):
            duplicate_picks[gameid] += 1
        if len(bans) != len(set(bans)):
            duplicate_bans[gameid] += 1
        if set(picks).intersection(bans):
            pick_ban_overlap[gameid] += 1

    team_rows = len(team_df)
    games = int(team_df["gameid"].nunique()) if "gameid" in team_df.columns else 0
    team_rows_per_game = team_df.groupby("gameid").size()
    incomplete_games = team_rows_per_game[team_rows_per_game != 2]

    side_sets = (
        team_df.groupby("gameid")["side"]
        .apply(lambda values: {str(value).lower() for value in values if pd.notna(value)})
    )
    missing_side_games = side_sets[side_sets != {"blue", "red"}]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(csv_path),
        "summary": {
            "total_rows": int(len(teams_df)),
            "team_rows": int(team_rows),
            "games": games,
            "games_with_single_team": int(incomplete_games.count()),
            "games_with_missing_side": int(missing_side_games.count()),
        },
        "missing_picks": {
            "total": int(sum(missing_pick_by_col.values())),
            "by_column": counter_payload(missing_pick_by_col),
            "by_side": counter_payload(missing_pick_by_side),
            "by_league": counter_payload(missing_pick_by_league, top_n),
            "top_games": counter_payload(missing_pick_by_game, top_n),
        },
        "missing_bans": {
            "total": int(sum(missing_ban_by_col.values())),
            "by_column": counter_payload(missing_ban_by_col),
            "by_side": counter_payload(missing_ban_by_side),
            "by_league": counter_payload(missing_ban_by_league, top_n),
            "top_games": counter_payload(missing_ban_by_game, top_n),
            "games_with_all_bans_missing": counter_payload(all_bans_missing_games, top_n),
        },
        "unknown_champions": {
            "pick_names": counter_payload(unknown_pick_names, top_n),
            "ban_names": counter_payload(unknown_ban_names, top_n),
            "mapping_loaded": known_champions is not None,
        },
        "duplicate_picks": {
            "games": int(len(duplicate_picks)),
            "top_games": counter_payload(duplicate_picks, top_n),
        },
        "duplicate_bans": {
            "games": int(len(duplicate_bans)),
            "top_games": counter_payload(duplicate_bans, top_n),
        },
        "pick_ban_overlap": {
            "games": int(len(pick_ban_overlap)),
            "top_games": counter_payload(pick_ban_overlap, top_n),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report on data-quality issues in processed pro data.")
    parser.add_argument(
        "--input-dir",
        default="data/prodata",
        help="Directory containing lol-pro-data-processor outputs.",
    )
    parser.add_argument(
        "--champion-mapping-path",
        default=DEFAULT_MAPPING_PATH,
        help="Path to the DDragon champion mapping artifact JSON.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the report as JSON.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top entries to include for counters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.input_dir, args.champion_mapping_path, args.top_n)
    payload = json.dumps(report, indent=2, sort_keys=True)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
