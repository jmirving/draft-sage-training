from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from draft_sage_training.utils.champion_sanitizer import ChampionSanitizer
from draft_sage_training.utils.champion_mapping import load_champion_mapping
from draft_sage_training.utils.draft_order import DRAFT_ORDER


PICK_COLUMNS = ["pick1", "pick2", "pick3", "pick4", "pick5"]
BAN_COLUMNS = ["ban1", "ban2", "ban3", "ban4", "ban5"]
TEAM_IDS = {100, 200}


def load_latest_csv(directory: Path, patterns: Sequence[str]) -> Path:
    for pattern in patterns:
        candidates = list(directory.glob(pattern))
        if not candidates:
            continue
        return max(candidates, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"No CSV files found in {directory}")


def load_processed_teams(input_dir: str) -> pd.DataFrame:
    input_path = Path(input_dir)
    teams_dir = input_path / "teams" if (input_path / "teams").is_dir() else input_path
    latest_csv = load_latest_csv(teams_dir, ["teams_*.csv", "*.csv"])
    return pd.read_csv(latest_csv)


def drop_incomplete_team_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe

    if "participantid" not in dataframe.columns:
        return dataframe

    team_mask = dataframe["participantid"].isin(list(TEAM_IDS))
    if not team_mask.any():
        return dataframe

    rows_to_drop = pd.Series(False, index=dataframe.index)

    picks_only = dataframe.loc[team_mask, PICK_COLUMNS].copy()
    for column in PICK_COLUMNS:
        picks_only[column] = picks_only[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    missing_picks_mask = picks_only.isna() | (picks_only == "")
    has_missing_pick = missing_picks_mask.any(axis=1)
    missing_pick_rows = team_mask & has_missing_pick
    if missing_pick_rows.any():
        logging.info("Dropping %d team rows with incomplete picks.", int(missing_pick_rows.sum()))
        rows_to_drop |= missing_pick_rows

    ban_columns = [column for column in BAN_COLUMNS if column in dataframe.columns]
    if ban_columns:
        bans_only = dataframe.loc[team_mask, ban_columns].copy()
        for column in ban_columns:
            bans_only[column] = bans_only[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )
        missing_bans_mask = bans_only.isna() | (bans_only == "")
        all_missing_bans = missing_bans_mask.all(axis=1)
        missing_ban_rows = team_mask & all_missing_bans
        if missing_ban_rows.any():
            logging.info("Dropping %d team rows with all bans missing.", int(missing_ban_rows.sum()))
            rows_to_drop |= missing_ban_rows

    if rows_to_drop.any():
        return dataframe.loc[~rows_to_drop].copy()
    return dataframe


def sort_patch_values(patches: Iterable[str]) -> list[str]:
    patches_list = [str(patch) for patch in patches if patch]
    if not patches_list:
        return []
    try:
        numeric = [float(patch) for patch in patches_list]
    except ValueError:
        return sorted(patches_list)
    paired = sorted(zip(numeric, patches_list), key=lambda item: item[0])
    return [patch for _, patch in paired]


def filter_patches(
    dataframe: pd.DataFrame,
    patch_window: Optional[int] = None,
    patches: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty or "patch" not in dataframe.columns:
        return dataframe

    normalized = dataframe.copy()
    normalized["patch"] = normalized["patch"].astype(str)

    if patches:
        allowed = {str(patch) for patch in patches}
        return normalized[normalized["patch"].isin(allowed)].copy()

    if patch_window:
        unique_patches = sort_patch_values(normalized["patch"].dropna().unique())
        if not unique_patches:
            return normalized
        window = set(unique_patches[-patch_window:])
        return normalized[normalized["patch"].isin(window)].copy()

    return normalized


def detect_fearless_series(
    dataframe: pd.DataFrame,
    sanitizer: ChampionSanitizer,
) -> dict[str, bool]:
    # Heuristic: fearless series should not repeat picks across games and should not
    # ban champions that were picked earlier in the series.
    if dataframe is None or dataframe.empty:
        return {}

    required = {"seriesid", "gameid"}
    if not required.issubset(set(dataframe.columns)):
        return {}

    series_fearless: dict[str, bool] = {}
    for seriesid, series_rows in dataframe.groupby("seriesid"):
        game_ids = series_rows["gameid"].dropna().unique()
        if len(game_ids) < 2:
            series_fearless[seriesid] = False
            continue

        seen_picks: set[str] = set()
        repeated_pick = False
        banned_previous_pick = False

        ordering_column = "game" if "game" in series_rows.columns else "gameid"
        ordered_rows = series_rows.sort_values(ordering_column)

        for _, game_rows in ordered_rows.groupby("gameid", sort=False):
            game_picks: set[str] = set()
            for pick_col in PICK_COLUMNS:
                if pick_col not in game_rows.columns:
                    continue
                for value in game_rows[pick_col].tolist():
                    if pd.isna(value) or value == "":
                        continue
                    sanitized = sanitizer.sanitize(value)
                    if sanitized:
                        game_picks.add(sanitized)

            if seen_picks.intersection(game_picks):
                repeated_pick = True
                break

            game_bans: set[str] = set()
            for ban_col in BAN_COLUMNS:
                if ban_col not in game_rows.columns:
                    continue
                for value in game_rows[ban_col].tolist():
                    if pd.isna(value) or value == "":
                        continue
                    sanitized = sanitizer.sanitize(value)
                    if sanitized:
                        game_bans.add(sanitized)

            if seen_picks.intersection(game_bans):
                banned_previous_pick = True
                break

            seen_picks.update(game_picks)

        series_fearless[seriesid] = not (repeated_pick or banned_previous_pick)

    return series_fearless


def infer_series_ids(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return dataframe

    required = {"league", "split", "year", "gameid", "teamid", "game"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns for series inference: {sorted(missing)}")

    sort_columns = ["league", "split", "year"]
    if "date" in dataframe.columns:
        sort_columns.append("date")
    sort_columns.extend(["gameid", "teamid", "game"])

    dataframe = dataframe.sort_values(sort_columns).reset_index(drop=True)
    parsed_dates = pd.to_datetime(dataframe.get("date"), errors="coerce") if "date" in dataframe.columns else None

    series_counters: dict[tuple, int] = {}
    last_game_numbers: dict[tuple, int] = {}
    last_dates: dict[tuple, object] = {}

    rows_to_keep = []
    series_ids: list[str] = []
    drop_gameids = set()

    for idx, row in dataframe.iterrows():
        gameid = row["gameid"]
        teamid = row["teamid"]
        opponent_rows = dataframe[(dataframe["gameid"] == gameid) & (dataframe["teamid"] != teamid)]
        if opponent_rows.empty:
            if gameid not in drop_gameids:
                logging.warning("Missing opponent data for gameid=%s; dropping game", gameid)
            drop_gameids.add(gameid)
            continue

        other_teamid = opponent_rows["teamid"].iloc[0]
        matchup = tuple(sorted([str(teamid), str(other_teamid)]))
        key = (row["league"], row["split"], row["year"], matchup)

        game_number = row["game"]
        current_date = None
        if parsed_dates is not None:
            parsed_value = parsed_dates.iloc[idx]
            current_date = parsed_value.date() if not pd.isna(parsed_value) else None

        if (
            key not in series_counters
            or game_number < last_game_numbers.get(key, 0)
            or (
                current_date is not None
                and last_dates.get(key) is not None
                and current_date != last_dates.get(key)
            )
        ):
            series_counters[key] = series_counters.get(key, 0) + 1

        series_id = f"{row['league']}_{row['split']}_{row['year']}_{matchup[0]}_{matchup[1]}_S{series_counters[key]}"
        rows_to_keep.append(idx)
        series_ids.append(series_id)
        last_game_numbers[key] = game_number
        if current_date is not None:
            last_dates[key] = current_date

    dataframe = dataframe.loc[rows_to_keep].copy()
    dataframe["seriesid"] = series_ids
    dataframe = dataframe[~dataframe["gameid"].isin(drop_gameids)].reset_index(drop=True)
    return dataframe


def aggregate_training_data(
    teams_df: pd.DataFrame,
    patch_window: Optional[int] = None,
    patches: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if teams_df is None:
        return teams_df

    dataframe = teams_df.copy()
    if "participantid" in dataframe.columns:
        dataframe = dataframe[dataframe["participantid"].isin(list(TEAM_IDS))].copy()

    dataframe = drop_incomplete_team_rows(dataframe)
    dataframe = filter_patches(dataframe, patch_window=patch_window, patches=patches)
    dataframe = infer_series_ids(dataframe)

    sort_cols = [col for col in ["seriesid", "gameid", "eventid", "pickbannumber", "order"] if col in dataframe.columns]
    if sort_cols:
        dataframe = dataframe.sort_values(sort_cols).reset_index(drop=True)
    return dataframe


class DraftDataset(Dataset):
    def __init__(
        self,
        input_dir: Optional[str] = None,
        teams_df: Optional[pd.DataFrame] = None,
        patch_window: Optional[int] = None,
        patches: Optional[Sequence[str]] = None,
        champion_mapping_path: Optional[str] = None,
    ):
        if teams_df is None:
            if input_dir is None:
                raise ValueError("input_dir or teams_df is required")
            teams_df = load_processed_teams(input_dir)

        self.data = aggregate_training_data(teams_df, patch_window=patch_window, patches=patches)
        self.champion_sanitizer = ChampionSanitizer()
        self.champion2idx = {}
        self.idx2champion = {}

        self.missing_key = self.champion_sanitizer.sanitize("MISSING")
        self.champion2idx[self.missing_key] = 0
        self.idx2champion[0] = self.missing_key

        mapping_entries = load_champion_mapping(champion_mapping_path)
        real_champions = [
            entry.get("normalized_name")
            for entry in mapping_entries
            if entry.get("normalized_name")
        ]
        for i, champ_name in enumerate(real_champions, start=1):
            sanitized_name = self.champion_sanitizer.sanitize(champ_name)
            self.champion2idx[sanitized_name] = i
            self.idx2champion[i] = sanitized_name

        self.num_champions = len(self.champion2idx)
        self.draft_features = 20

        self.series_fearless = detect_fearless_series(self.data, self.champion_sanitizer)
        if self.series_fearless:
            fearless_count = sum(1 for value in self.series_fearless.values() if value)
            logging.info(
                "Detected %d fearless series out of %d total series.",
                fearless_count,
                len(self.series_fearless),
            )

        self.patch_values = sort_patch_values(self.data["patch"].dropna().unique()) if "patch" in self.data.columns else []
        self.patch_to_index = {patch: index + 1 for index, patch in enumerate(self.patch_values)}
        self.unknown_patch_index = 0
        self.num_patches = len(self.patch_values) + 1

        self.samples = self._preprocess_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        output_mask = self.get_output_mask(row["already_picked_or_banned"])
        target = row["target"] - 1 if row["target"] > 0 else 0
        return {
            "draft_sequence": torch.tensor(row["draft_sequence"], dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "output_mask": torch.tensor(output_mask, dtype=torch.float32),
            "patch_index": torch.tensor(row["patch_index"], dtype=torch.long),
        }

    def get_output_mask(self, already_picked_or_banned):
        mask = np.ones(self.num_champions - 1, dtype=np.float32)
        for champ in already_picked_or_banned:
            idx = self.champion2idx.get(champ)
            if idx is not None and idx > 0:
                mask[idx - 1] = 0
        return mask

    def _normalize_champion_id(self, champion_name: str) -> int:
        if pd.isna(champion_name) or champion_name == "nan":
            champion_name = "MISSING"

        champion_name = self.champion_sanitizer.sanitize(champion_name)
        if not champion_name:
            champion_name = self.missing_key

        return self.champion2idx.get(champion_name, self.champion2idx[self.missing_key])

    def _patch_index_for_rows(self, blue_row: pd.Series, red_row: pd.Series) -> int:
        patch_value = blue_row.get("patch") if blue_row is not None else None
        if patch_value is None or pd.isna(patch_value):
            patch_value = red_row.get("patch") if red_row is not None else None
        if patch_value is None or pd.isna(patch_value):
            return self.unknown_patch_index
        patch_str = str(patch_value)
        return self.patch_to_index.get(patch_str, self.unknown_patch_index)

    def _preprocess_samples(self):
        samples = []
        grouped_games = self.data.groupby(["seriesid", "gameid"])

        for (seriesid, gameid), game_rows in grouped_games:
            blue_rows = game_rows[game_rows["side"].str.lower() == "blue"]
            red_rows = game_rows[game_rows["side"].str.lower() == "red"]

            if blue_rows.empty or red_rows.empty:
                logging.warning(
                    "Missing blue or red side data for series %s game %s; skipping",
                    seriesid,
                    gameid,
                )
                continue

            blue_row = blue_rows.iloc[0]
            red_row = red_rows.iloc[0]
            patch_index = self._patch_index_for_rows(blue_row, red_row)

            fearless_picks = set()
            if self.series_fearless.get(seriesid, False):
                game_number = blue_row.get("game") if "game" in blue_row else None
                if game_number is None or pd.isna(game_number):
                    previous_games = self.data[
                        (self.data["seriesid"] == seriesid) & (self.data["gameid"] < gameid)
                    ]
                else:
                    previous_games = self.data[
                        (self.data["seriesid"] == seriesid) & (self.data["game"] < game_number)
                    ]
                for _, prev_row in previous_games.iterrows():
                    for pick_number in range(1, 6):
                        pick_col = f"pick{pick_number}"
                        if pick_col in prev_row and not pd.isna(prev_row[pick_col]):
                            sanitized_pick = self.champion_sanitizer.sanitize(prev_row[pick_col])
                            if sanitized_pick:
                                fearless_picks.add(sanitized_pick)

            used_champions = set(fearless_picks)
            draft_sequence = [0] * self.draft_features

            for event_index, (side, action_type, action_number) in enumerate(DRAFT_ORDER):
                row = blue_row if side == "blue" else red_row
                column_prefix = "ban" if action_type == "ban" else "pick"
                column_name = f"{column_prefix}{action_number}"
                champion_name = row.get(column_name)
                champion_index = self._normalize_champion_id(champion_name)
                if champion_index == 0:
                    logging.warning(
                        "Missing or unknown champion for %s %s %s in series %s game %s; skipping sample",
                        side,
                        action_type,
                        action_number,
                        seriesid,
                        gameid,
                    )
                    continue

                sanitized_name = self.champion_sanitizer.sanitize(champion_name)
                if sanitized_name in used_champions:
                    logging.warning(
                        "Duplicate %s %s %s (%s) in series %s game %s; skipping sample",
                        side,
                        action_type,
                        action_number,
                        sanitized_name,
                        seriesid,
                        gameid,
                    )
                    continue

                samples.append(
                    {
                        "draft_sequence": draft_sequence.copy(),
                        "target": champion_index,
                        "already_picked_or_banned": set(used_champions),
                        "patch_index": patch_index,
                    }
                )

                if sanitized_name:
                    used_champions.add(sanitized_name)

                draft_sequence[event_index] = champion_index

        return samples
