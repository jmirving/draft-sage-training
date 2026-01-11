#!/usr/bin/env python3
"""Generate per-patch champion prior weights from processed team data."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from draft_sage_training.dataset import (
    BAN_COLUMNS,
    PICK_COLUMNS,
    TEAM_IDS,
    filter_patches,
    load_processed_teams,
    sort_patch_values,
)
from draft_sage_training.utils.champion_mapping import load_champion_mapping
from draft_sage_training.utils.champion_sanitizer import ChampionSanitizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-patch champion priors from picks/bans.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Processed prodata directory (or direct teams CSV folder).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/weights/champion-priors",
        help="Output directory for per-patch weight artifacts.",
    )
    parser.add_argument(
        "--champion-mapping-path",
        required=True,
        help="Path to champion-mapping latest.json artifact.",
    )
    parser.add_argument(
        "--patch-window",
        type=int,
        default=None,
        help="Optional patch window (most recent N patches).",
    )
    parser.add_argument(
        "--patch",
        dest="patches",
        action="append",
        default=None,
        help="Explicit patch to include; repeatable.",
    )
    parser.add_argument(
        "--no-bans",
        action="store_true",
        help="Exclude bans from the prior calculation.",
    )
    parser.add_argument(
        "--no-picks",
        action="store_true",
        help="Exclude picks from the prior calculation.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Meta source label (defaults to input-dir name).",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional meta note to include in the output.",
    )
    return parser.parse_args()


def drop_incomplete_team_rows(
    dataframe: pd.DataFrame,
    include_picks: bool = True,
    include_bans: bool = True,
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe

    if "participantid" not in dataframe.columns:
        return dataframe

    team_mask = dataframe["participantid"].isin(list(TEAM_IDS))
    if not team_mask.any():
        return dataframe

    rows_to_drop = pd.Series(False, index=dataframe.index)

    if include_picks:
        picks_only = dataframe.loc[team_mask, PICK_COLUMNS].copy()
        for column in PICK_COLUMNS:
            if column not in picks_only.columns:
                continue
            picks_only[column] = picks_only[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )
        missing_picks_mask = picks_only.isna() | (picks_only == "")
        has_missing_pick = missing_picks_mask.any(axis=1)
        missing_pick_rows = team_mask & has_missing_pick
        if missing_pick_rows.any():
            logging.info("Dropping %d team rows with incomplete picks.", int(missing_pick_rows.sum()))
            rows_to_drop |= missing_pick_rows

    if include_bans:
        ban_columns = [column for column in BAN_COLUMNS if column in dataframe.columns]
        if ban_columns:
            bans_only = dataframe.loc[team_mask, ban_columns].copy()
            for column in ban_columns:
                bans_only[column] = bans_only[column].apply(
                    lambda value: value.strip() if isinstance(value, str) else value
                )
            missing_bans_mask = bans_only.isna() | (bans_only == "")
            missing_any_ban = missing_bans_mask.any(axis=1)
            missing_ban_rows = team_mask & missing_any_ban
            if missing_ban_rows.any():
                logging.info("Dropping %d team rows with missing bans.", int(missing_ban_rows.sum()))
                rows_to_drop |= missing_ban_rows

    if rows_to_drop.any():
        return dataframe.loc[~rows_to_drop].copy()
    return dataframe


def drop_incomplete_games(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe

    if "gameid" not in dataframe.columns or "side" not in dataframe.columns:
        return dataframe

    team_df = dataframe
    if "participantid" in dataframe.columns:
        team_mask = dataframe["participantid"].isin(list(TEAM_IDS))
        if team_mask.any():
            team_df = dataframe.loc[team_mask].copy()

    valid_game_ids = []
    for gameid, rows in team_df.groupby("gameid"):
        if len(rows) != 2:
            continue
        sides = {str(value).lower() for value in rows["side"] if pd.notna(value)}
        if sides == {"blue", "red"}:
            valid_game_ids.append(gameid)

    total_games = team_df["gameid"].nunique()
    dropped_games = total_games - len(valid_game_ids)
    if dropped_games > 0:
        logging.info("Dropping %d games without complete blue/red team rows.", dropped_games)

    return dataframe[dataframe["gameid"].isin(valid_game_ids)].copy()


def build_champion_lookup(
    mapping_entries: Iterable[dict[str, object]],
    sanitizer: ChampionSanitizer,
) -> dict[str, str]:
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


def normalize_weights(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {champ: count / total for champ, count in counts.items()}


def write_weights(
    output_dir: Path,
    patch_value: str,
    weights: dict[str, float],
    meta: dict[str, object],
) -> None:
    payload = {
        "schema_version": "1.0",
        "weight_type": "champion-priors",
        "patch": patch_value,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "normalization": "sum-to-1",
        "weights": dict(sorted(weights.items())),
        "meta": meta,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{patch_value}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    include_picks = not args.no_picks
    include_bans = not args.no_bans

    if not include_picks and not include_bans:
        raise ValueError("At least one of picks or bans must be included.")

    if args.patch_window and args.patches:
        raise ValueError("Provide either --patch-window or --patch, not both.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    teams_df = load_processed_teams(args.input_dir)
    if "participantid" in teams_df.columns:
        teams_df = teams_df[teams_df["participantid"].isin(list(TEAM_IDS))].copy()

    teams_df = drop_incomplete_team_rows(
        teams_df,
        include_picks=include_picks,
        include_bans=include_bans,
    )
    teams_df = drop_incomplete_games(teams_df)
    teams_df = filter_patches(teams_df, patch_window=args.patch_window, patches=args.patches)

    if "patch" not in teams_df.columns:
        raise ValueError("Patch column missing from teams data.")

    teams_df = teams_df[teams_df["patch"].notna()].copy()
    teams_df["patch"] = teams_df["patch"].astype(str)
    teams_df = teams_df[teams_df["patch"].str.strip() != ""].copy()

    pick_columns = [col for col in PICK_COLUMNS if col in teams_df.columns] if include_picks else []
    ban_columns = [col for col in BAN_COLUMNS if col in teams_df.columns] if include_bans else []
    if not pick_columns and not ban_columns:
        raise ValueError("No pick/ban columns found for the requested options.")

    mapping_entries = load_champion_mapping(args.champion_mapping_path)
    sanitizer = ChampionSanitizer()
    champion_lookup = build_champion_lookup(mapping_entries, sanitizer)
    if not champion_lookup:
        raise ValueError("Champion mapping did not contain any normalized entries.")

    patches = sort_patch_values(teams_df["patch"].dropna().unique())
    output_dir = Path(args.output_dir)
    source_label = args.source or Path(args.input_dir).name

    unknown_counts: Counter[str] = Counter()
    total_written = 0

    for patch_value in patches:
        patch_df = teams_df[teams_df["patch"] == patch_value]
        counts: Counter[str] = Counter()

        for _, row in patch_df.iterrows():
            for column in pick_columns + ban_columns:
                value = row.get(column)
                if pd.isna(value) or value == "":
                    continue
                sanitized = sanitizer.sanitize(value)
                if not sanitized:
                    continue
                normalized = champion_lookup.get(sanitized)
                if not normalized:
                    unknown_counts[sanitized] += 1
                    continue
                counts[normalized] += 1

        weights = normalize_weights(counts)
        if not weights:
            logging.warning("No weights generated for patch %s; skipping.", patch_value)
            continue

        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Weights for patch {patch_value} do not sum to 1 (got {weight_sum}).")

        meta = {
            "source": source_label,
            "window": "per-patch",
            "picks_included": include_picks,
            "bans_included": include_bans,
            "fearless_included": True,
            "rows": int(len(patch_df)),
        }
        if args.patch_window:
            meta["patch_window"] = args.patch_window
        if args.patches:
            meta["patches"] = args.patches
        if args.note:
            meta["note"] = args.note

        write_weights(output_dir, patch_value, weights, meta)
        total_written += 1

    if unknown_counts:
        logging.warning(
            "Unknown champions encountered (top 10 shown): %s",
            unknown_counts.most_common(10),
        )

    logging.info("Wrote %d patch files to %s", total_written, output_dir)


if __name__ == "__main__":
    main()
