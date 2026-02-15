#!/usr/bin/env python3
"""Generate causal, patch-weighted role priors from player role records."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from draft_sage_training.dataset import TEAM_IDS, filter_patches, load_latest_csv, sort_patch_values
from draft_sage_training.utils.champion_mapping import load_champion_mapping
from draft_sage_training.utils.champion_sanitizer import ChampionSanitizer
from draft_sage_training.utils.role_priors import (
    DEFAULT_ROLE_ORDER,
    build_causal_patch_weights,
)


ROLE_ALIASES = {
    "top": "top",
    "jng": "jungle",
    "jg": "jungle",
    "jungle": "jungle",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "ad": "bot",
    "carry": "bot",
    "sup": "support",
    "support": "support",
}
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / ".tmp" / "prodata-2025-plus-2026-01-clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-patch role priors from team picks.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=(
            "Processed prodata directory with players CSVs. "
            "Defaults to cleaned 2025 + Jan 2026 dataset."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="data/weights/role-priors",
        help="Output directory for per-patch role prior artifacts.",
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
        "--source",
        default=None,
        help="Meta source label (defaults to input-dir name).",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Optional meta note to include in the output.",
    )
    parser.add_argument(
        "--latest-major-weight",
        type=float,
        default=4.0,
        help="Multiplier for patches in the same major patch family as the target patch.",
    )
    parser.add_argument(
        "--patch-recency-decay",
        type=float,
        default=0.9,
        help="Decay factor for older patches (weight *= decay**distance).",
    )
    parser.add_argument(
        "--older-major-decay",
        type=float,
        default=0.35,
        help="Per-major-step decay for older major patches.",
    )
    return parser.parse_args()


def build_champion_lookup(mapping_entries: list[dict[str, object]], sanitizer: ChampionSanitizer) -> dict[str, str]:
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


def load_processed_players(input_dir: str) -> pd.DataFrame:
    input_path = Path(input_dir)
    players_dir = input_path / "players" if (input_path / "players").is_dir() else input_path
    player_files = sorted(players_dir.glob("players_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not player_files:
        raise FileNotFoundError(
            f"No players_*.csv found in {players_dir}. "
            "Role priors require processed player rows."
        )

    required_columns = {"patch", "position", "champion"}
    selected_file: Path | None = None
    for candidate in player_files:
        try:
            with candidate.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header = next(reader, [])
        except OSError:
            continue
        if required_columns.issubset(set(header)):
            selected_file = candidate
            break

    if selected_file is None:
        selected_file = player_files[0]

    return pd.read_csv(selected_file)


def normalize_role(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    return ROLE_ALIASES.get(normalized)


def normalize_role_counts(
    counts: dict[str, Counter],
    champions: list[str],
    roles: list[str],
) -> dict[str, dict[str, float]]:
    weights: dict[str, dict[str, float]] = {}
    uniform = 1.0 / len(roles)
    for champion in champions:
        role_counts = counts.get(champion)
        if not role_counts:
            weights[champion] = {role: uniform for role in roles}
            continue
        total = sum(role_counts.values())
        if total <= 0:
            weights[champion] = {role: uniform for role in roles}
            continue
        weights[champion] = {role: role_counts.get(role, 0) / total for role in roles}
    return weights


def build_patch_role_counts(
    players_df: pd.DataFrame,
    champion_lookup: dict[str, str],
    sanitizer: ChampionSanitizer,
    patches: list[str],
    unknown_counts: Counter[str],
) -> dict[str, dict[str, Counter]]:
    counts_by_patch: dict[str, dict[str, Counter]] = {}
    for patch_value in patches:
        patch_df = players_df[players_df["patch"] == patch_value]
        counts: dict[str, Counter] = defaultdict(Counter)

        for _, row in patch_df.iterrows():
            role_value = normalize_role(row.get("position"))
            if not role_value:
                continue
            champ_value = row.get("champion")
            if pd.isna(champ_value) or champ_value == "":
                continue
            sanitized = sanitizer.sanitize(champ_value)
            if not sanitized:
                continue
            normalized = champion_lookup.get(sanitized)
            if not normalized:
                unknown_counts[sanitized] += 1
                continue
            counts[normalized][role_value] += 1

        counts_by_patch[patch_value] = counts

    return counts_by_patch


def aggregate_weighted_counts(
    counts_by_patch: dict[str, dict[str, Counter]],
    patch_weights: dict[str, float],
) -> dict[str, Counter]:
    aggregated: dict[str, Counter] = defaultdict(Counter)
    for patch_value, weight in patch_weights.items():
        if weight <= 0:
            continue
        source_counts = counts_by_patch.get(patch_value)
        if not source_counts:
            continue
        for champion, role_counts in source_counts.items():
            for role, count in role_counts.items():
                aggregated[champion][role] += float(count) * weight
    return aggregated


def write_weights(
    output_dir: Path,
    patch_value: str,
    weights: dict[str, dict[str, float]],
    meta: dict[str, object],
) -> None:
    payload = {
        "schema_version": "1.0",
        "weight_type": "role-priors",
        "patch": patch_value,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "normalization": "sum-to-1",
        "roles": DEFAULT_ROLE_ORDER,
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
    if args.patch_window and args.patches:
        raise ValueError("Provide either --patch-window or --patch, not both.")
    if args.latest_major_weight <= 0:
        raise ValueError("--latest-major-weight must be > 0.")
    if args.patch_recency_decay <= 0:
        raise ValueError("--patch-recency-decay must be > 0.")
    if args.older_major_decay <= 0:
        raise ValueError("--older-major-decay must be > 0.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    players_df = load_processed_players(args.input_dir)
    if "participantid" in players_df.columns:
        players_df = players_df[~players_df["participantid"].isin(list(TEAM_IDS))].copy()

    players_df = filter_patches(players_df, patch_window=args.patch_window, patches=args.patches)

    if "patch" not in players_df.columns:
        raise ValueError("Patch column missing from player data.")
    if "position" not in players_df.columns:
        raise ValueError("Position column missing from player data.")
    if "champion" not in players_df.columns:
        raise ValueError("Champion column missing from player data.")

    players_df = players_df[players_df["patch"].notna()].copy()
    players_df["patch"] = players_df["patch"].astype(str)
    players_df = players_df[players_df["patch"].str.strip() != ""].copy()

    mapping_entries = load_champion_mapping(args.champion_mapping_path)
    sanitizer = ChampionSanitizer()
    champion_lookup = build_champion_lookup(mapping_entries, sanitizer)
    if not champion_lookup:
        raise ValueError("Champion mapping did not contain any normalized entries.")

    champion_names = sorted({value for value in champion_lookup.values()})
    patches = sort_patch_values(players_df["patch"].dropna().unique())
    output_dir = Path(args.output_dir)
    source_label = args.source or Path(args.input_dir).name

    unknown_counts: Counter[str] = Counter()
    total_written = 0
    patch_rows = {patch_value: int((players_df["patch"] == patch_value).sum()) for patch_value in patches}
    cumulative_rows = 0
    cumulative_rows_by_patch: dict[str, int] = {}
    for patch_value in patches:
        cumulative_rows += patch_rows[patch_value]
        cumulative_rows_by_patch[patch_value] = cumulative_rows

    counts_by_patch = build_patch_role_counts(
        players_df,
        champion_lookup,
        sanitizer,
        patches,
        unknown_counts,
    )

    for target_index, patch_value in enumerate(patches):
        patch_weights = build_causal_patch_weights(
            patches,
            target_index,
            latest_major_weight=args.latest_major_weight,
            patch_recency_decay=args.patch_recency_decay,
            older_major_decay=args.older_major_decay,
        )
        counts = aggregate_weighted_counts(counts_by_patch, patch_weights)
        weights = normalize_role_counts(counts, champion_names, DEFAULT_ROLE_ORDER)

        meta = {
            "source": source_label,
            "window": "causal-patch-weighted",
            "rows_current_patch": patch_rows[patch_value],
            "rows_causal_history": cumulative_rows_by_patch[patch_value],
            "role_field": "position",
            "patch_weighting": {
                "latest_major_weight": args.latest_major_weight,
                "patch_recency_decay": args.patch_recency_decay,
                "older_major_decay": args.older_major_decay,
                "causal": True,
            },
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
