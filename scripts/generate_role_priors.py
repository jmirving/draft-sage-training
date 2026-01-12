#!/usr/bin/env python3
"""Generate per-patch role priors from player role records."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from draft_sage_training.dataset import TEAM_IDS, filter_patches, load_latest_csv, sort_patch_values
from draft_sage_training.utils.champion_mapping import load_champion_mapping
from draft_sage_training.utils.champion_sanitizer import ChampionSanitizer
from draft_sage_training.utils.role_priors import DEFAULT_ROLE_ORDER


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
DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parents[2] / "draft-sage" / "data" / "processed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-patch role priors from team picks.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Processed prodata directory with players CSVs.",
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
    latest_csv = load_latest_csv(players_dir, ["players_*.csv", "*.csv"])
    return pd.read_csv(latest_csv)


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

        weights = normalize_role_counts(counts, champion_names, DEFAULT_ROLE_ORDER)

        meta = {
            "source": source_label,
            "window": "per-patch",
            "rows": int(len(patch_df)),
            "role_field": "position",
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
