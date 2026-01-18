from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from draft_sage_training.utils.champion_sanitizer import ChampionSanitizer
from draft_sage_training.utils.champion_eligibility import load_champion_eligibility
from draft_sage_training.utils.champion_mapping import load_champion_mapping
from draft_sage_training.utils.draft_order import DRAFT_ORDER
from draft_sage_training.utils.role_priors import DEFAULT_ROLE_ORDER, validate_role_priors_payload


PICK_COLUMNS = ["pick1", "pick2", "pick3", "pick4", "pick5"]
BAN_COLUMNS = ["ban1", "ban2", "ban3", "ban4", "ban5"]
TEAM_IDS = {100, 200}
EARLY_BLUE_BAN_MAX = 3
EARLY_BLUE_BAN_PRIOR_SERIES_WEIGHT = 0.6
EARLY_BLUE_BAN_PRIOR_TEAM_WEIGHT = 0.4
TEAM_PRIOR_BAN_WEIGHT = 0.5
TEAM_PRIOR_PICK_WEIGHT = 0.5
SERIES_PRIOR_SAME_SIDE_WEIGHT = 0.55
SERIES_PRIOR_OVERALL_WEIGHT = 0.25
SERIES_PRIOR_OPPOSITE_WEIGHT = 0.10
SERIES_PRIOR_MISS_WEIGHT = 0.10


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
    kept_games = len(valid_game_ids)
    dropped_games = total_games - kept_games
    if dropped_games > 0:
        logging.info("Dropping %d games without complete blue/red team rows.", dropped_games)

    return dataframe[dataframe["gameid"].isin(valid_game_ids)].copy()


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


def normalize_category(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


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
    dataframe = drop_incomplete_games(dataframe)
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
        champion_eligibility_path: Optional[str] = None,
        champion_priors_dir: Optional[str] = None,
        champion_priors_strength: float = 1.0,
        champion_priors_time_buckets: int = 1,
        role_priors_dir: Optional[str] = None,
        role_priors_strength: float = 1.0,
        early_blue_ban_priors: bool = False,
        early_blue_ban_priors_strength: float = 1.0,
        team_priors_window_days: int = 30,
        use_league_team_embeddings: bool = True,
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
            str(entry.get("normalized_name"))
            for entry in mapping_entries
            if entry.get("normalized_name")
        ]
        self.champion_names = real_champions
        for i, champ_name in enumerate(real_champions, start=1):
            sanitized_name = self.champion_sanitizer.sanitize(champ_name)
            self.champion2idx[sanitized_name] = i
            self.idx2champion[i] = sanitized_name

        self.num_champions = len(self.champion2idx)
        self.draft_features = 20

        self.champion_eligibility_by_league: dict[str, np.ndarray] = {}
        if champion_eligibility_path:
            self.champion_eligibility_by_league = load_champion_eligibility(
                champion_eligibility_path,
                self.champion_sanitizer,
                self.champion2idx,
            )

        self.use_league_team_embeddings = use_league_team_embeddings
        self.unknown_league_index = 0
        self.unknown_team_index = 0
        if self.use_league_team_embeddings:
            self.league_values, self.league_to_index = self._build_category_index("league")
            self.team_values, self.team_to_index = self._build_category_index("teamid")
            self.num_leagues = len(self.league_to_index) + 1
            self.num_teams = len(self.team_to_index) + 1
        else:
            self.league_values = []
            self.league_to_index = {}
            self.team_values = []
            self.team_to_index = {}
            self.num_leagues = 1
            self.num_teams = 1

        self.champion_priors_strength = champion_priors_strength
        self.champion_priors_time_buckets = max(int(champion_priors_time_buckets), 1)
        self.champion_priors_by_patch: Optional[dict[str, torch.Tensor]] = None
        self.default_champion_priors = torch.zeros(self.num_champions - 1, dtype=torch.float32)
        self.patch_time_boundaries: dict[str, list[pd.Timestamp]] = {}
        if self.champion_priors_time_buckets > 1:
            self.patch_time_boundaries = self._build_patch_time_boundaries()
        if champion_priors_dir:
            self.champion_priors_by_patch = self._load_champion_priors(champion_priors_dir)

        self.role_priors_strength = role_priors_strength
        self.role_priors_by_patch: Optional[dict[str, np.ndarray]] = None
        self.default_role_priors = np.full(
            (self.num_champions - 1, len(DEFAULT_ROLE_ORDER)),
            1.0 / len(DEFAULT_ROLE_ORDER),
            dtype=np.float32,
        )
        if role_priors_dir:
            self.role_priors_by_patch = self._load_role_priors(role_priors_dir)

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

        self.early_blue_ban_priors_enabled = early_blue_ban_priors
        self.early_blue_ban_priors_strength = early_blue_ban_priors_strength
        self.team_priors_window_days = max(int(team_priors_window_days), 1)
        self.early_blue_ban_priors_by_game: dict[tuple[str, object], torch.Tensor] = {}
        if self.early_blue_ban_priors_enabled:
            self.early_blue_ban_priors_by_game = self._build_early_blue_ban_priors()

        self.samples = self._preprocess_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        output_mask = self.get_output_mask(
            row["already_picked_or_banned"],
            league_key=row.get("league_key"),
            game_date_value=row.get("game_date_value"),
        )
        target = row["target"] - 1 if row["target"] > 0 else 0
        action_type = row.get("action_type", "ban")
        side = row.get("side", "blue")
        payload = {
            "draft_sequence": torch.tensor(row["draft_sequence"], dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
            "output_mask": torch.tensor(output_mask, dtype=torch.float32),
            "patch_index": torch.tensor(row["patch_index"], dtype=torch.long),
            "action_type": torch.tensor(1 if action_type == "pick" else 0, dtype=torch.long),
            "side": torch.tensor(1 if side == "red" else 0, dtype=torch.long),
            "event_index": torch.tensor(row.get("event_index", 0), dtype=torch.long),
            "league_index": torch.tensor(row.get("league_index", self.unknown_league_index), dtype=torch.long),
            "team_index": torch.tensor(row.get("team_index", self.unknown_team_index), dtype=torch.long),
        }
        champion_priors = None
        if self.champion_priors_by_patch is not None:
            priors_key = row.get("priors_key") or row.get("patch")
            champion_priors = self.champion_priors_by_patch.get(
                priors_key, self.default_champion_priors
            )
        if self.early_blue_ban_priors_enabled and row.get("is_early_blue_ban"):
            game_key = (row.get("seriesid"), row.get("gameid"))
            ban_priors = self.early_blue_ban_priors_by_game.get(game_key)
            if ban_priors is not None:
                if champion_priors is None:
                    champion_priors = ban_priors
                else:
                    champion_priors = champion_priors + ban_priors
        if champion_priors is not None:
            payload["champion_priors"] = champion_priors
        if "role_priors" in row:
            payload["role_priors"] = torch.tensor(row["role_priors"], dtype=torch.float32)
        return payload

    def get_output_mask(
        self,
        already_picked_or_banned,
        league_key: Optional[str] = None,
        game_date_value: Optional[int] = None,
    ):
        mask = np.ones(self.num_champions - 1, dtype=np.float32)
        for champ in already_picked_or_banned:
            idx = self.champion2idx.get(champ)
            if idx is not None and idx > 0:
                mask[idx - 1] = 0
        if self.champion_eligibility_by_league and league_key and game_date_value is not None:
            eligibility = self.champion_eligibility_by_league.get(league_key)
            if eligibility is not None:
                mask = mask * (game_date_value >= eligibility).astype(np.float32)
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

    def _patch_value_for_rows(self, blue_row: pd.Series, red_row: pd.Series) -> str | None:
        patch_value = blue_row.get("patch") if blue_row is not None else None
        if patch_value is None or pd.isna(patch_value):
            patch_value = red_row.get("patch") if red_row is not None else None
        if patch_value is None or pd.isna(patch_value):
            return None
        patch_str = str(patch_value).strip()
        return patch_str or None

    def _parse_date_value(self, value: object) -> Optional[pd.Timestamp]:
        if value is None or pd.isna(value):
            return None
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed

    def _build_patch_time_boundaries(self) -> dict[str, list[pd.Timestamp]]:
        if "date" not in self.data.columns:
            raise ValueError("Time-aware priors require a date column in the dataset.")
        normalized = self.data.copy()
        normalized["_parsed_date"] = pd.to_datetime(normalized["date"], errors="coerce")
        normalized = normalized[normalized["_parsed_date"].notna()].copy()
        boundaries: dict[str, list[pd.Timestamp]] = {}
        for patch_value, rows in normalized.groupby("patch"):
            rows = rows.sort_values("_parsed_date")
            total = len(rows)
            if total == 0:
                continue
            cuts = []
            for bucket in range(1, self.champion_priors_time_buckets):
                idx = int(total * bucket / self.champion_priors_time_buckets)
                idx = min(max(idx, 1), total - 1)
                cuts.append(rows["_parsed_date"].iloc[idx])
            boundaries[str(patch_value)] = cuts
        return boundaries

    def _time_bucket_for_patch_date(self, patch_value: Optional[str], date_value: Optional[pd.Timestamp]) -> Optional[int]:
        if not patch_value or date_value is None:
            return None
        cuts = self.patch_time_boundaries.get(patch_value)
        if not cuts:
            return None
        for idx, boundary in enumerate(cuts):
            if date_value <= boundary:
                return idx
        return len(cuts)

    def _priors_key(self, patch_value: Optional[str], time_bucket: Optional[int]) -> Optional[str]:
        if not patch_value:
            return None
        if self.champion_priors_time_buckets <= 1 or time_bucket is None:
            return patch_value
        return f"{patch_value}__t{time_bucket + 1}of{self.champion_priors_time_buckets}"

    def _build_category_index(self, column: str) -> tuple[list[str], dict[str, int]]:
        if column not in self.data.columns:
            return [], {}
        values = []
        for raw_value in self.data[column].dropna().unique():
            normalized = normalize_category(raw_value)
            if normalized:
                values.append(normalized)
        unique_values = sorted(set(values))
        index = {value: idx + 1 for idx, value in enumerate(unique_values)}
        return unique_values, index

    def _league_index(self, value: object) -> int:
        if not self.use_league_team_embeddings:
            return self.unknown_league_index
        normalized = normalize_category(value)
        if not normalized:
            return self.unknown_league_index
        return self.league_to_index.get(normalized, self.unknown_league_index)

    def _team_index(self, value: object) -> int:
        if not self.use_league_team_embeddings:
            return self.unknown_team_index
        normalized = normalize_category(value)
        if not normalized:
            return self.unknown_team_index
        return self.team_to_index.get(normalized, self.unknown_team_index)

    def _load_champion_priors(self, priors_dir: str) -> dict[str, torch.Tensor]:
        priors_path = Path(priors_dir)
        if not priors_path.exists():
            raise FileNotFoundError(f"Champion priors directory not found: {priors_path}")

        weight_files = sorted(priors_path.glob("*.json"))
        if not weight_files:
            raise FileNotFoundError(f"No champion priors JSON files found in {priors_path}")

        priors_by_patch: dict[str, torch.Tensor] = {}
        for weight_file in weight_files:
            with weight_file.open(encoding="utf-8") as handle:
                payload = json.load(handle)

            weight_type = payload.get("weight_type")
            if weight_type and weight_type != "champion-priors":
                raise ValueError(f"Unexpected weight_type in {weight_file}: {weight_type}")

            patch_value = payload.get("patch") or weight_file.stem
            patch_key = str(patch_value)
            weights = payload.get("weights")
            if not isinstance(weights, dict):
                raise ValueError(f"Invalid weights payload in {weight_file}")

            vector = np.zeros(self.num_champions - 1, dtype=np.float32)
            for name, weight in weights.items():
                if weight is None:
                    continue
                weight_value = float(weight)
                if weight_value < 0:
                    raise ValueError(f"Negative weight for {name} in {weight_file}")
                sanitized = self.champion_sanitizer.sanitize(name)
                idx = self.champion2idx.get(sanitized)
                if idx is None or idx == 0:
                    raise ValueError(f"Unknown champion in priors: {name}")
                vector[idx - 1] = weight_value

            total = float(vector.sum())
            if total <= 0:
                raise ValueError(f"Champion priors for patch {patch_key} are empty.")
            if abs(total - 1.0) > 1e-4:
                raise ValueError(f"Champion priors for patch {patch_key} do not sum to 1.")

            priors_by_patch[patch_key] = torch.tensor(
                vector * self.champion_priors_strength,
                dtype=torch.float32,
            )

        return priors_by_patch

    def _load_role_priors(self, priors_dir: str) -> dict[str, np.ndarray]:
        priors_path = Path(priors_dir)
        if not priors_path.exists():
            raise FileNotFoundError(f"Role priors directory not found: {priors_path}")

        weight_files = sorted(priors_path.glob("*.json"))
        if not weight_files:
            raise FileNotFoundError(f"No role priors JSON files found in {priors_path}")

        priors_by_patch: dict[str, np.ndarray] = {}
        for weight_file in weight_files:
            with weight_file.open(encoding="utf-8") as handle:
                payload = json.load(handle)

            roles = validate_role_priors_payload(payload, self.champion_names)
            patch_value = payload.get("patch") or weight_file.stem
            patch_key = str(patch_value)

            weights = payload.get("weights")
            if not isinstance(weights, dict):
                raise ValueError(f"Invalid weights payload in {weight_file}")

            matrix = np.zeros((self.num_champions - 1, len(roles)), dtype=np.float32)
            for name, role_weights in weights.items():
                sanitized = self.champion_sanitizer.sanitize(name)
                idx = self.champion2idx.get(sanitized)
                if idx is None or idx == 0:
                    raise ValueError(f"Unknown champion in role priors: {name}")
                matrix[idx - 1] = [float(role_weights[role]) for role in roles]

            priors_by_patch[patch_key] = matrix

        return priors_by_patch

    def _normalize_prior(self, weights: np.ndarray) -> np.ndarray:
        total = float(weights.sum())
        if total <= 0:
            return np.zeros_like(weights)
        return weights / total

    def _build_pick_prior(self, pick_counts: np.ndarray, availability_counts: np.ndarray) -> np.ndarray:
        rates = np.divide(
            pick_counts,
            availability_counts,
            out=np.zeros_like(pick_counts),
            where=availability_counts > 0,
        )
        return self._normalize_prior(rates)

    def _blend_priors(self, priors: list[tuple[np.ndarray | None, float]]) -> np.ndarray:
        combined = np.zeros(self.num_champions - 1, dtype=np.float32)
        for prior, weight in priors:
            if prior is None:
                continue
            combined += prior * weight
        total = float(combined.sum())
        if total <= 0:
            return combined
        return combined / total

    def _game_sort_key(self, blue_row: pd.Series, red_row: pd.Series, gameid: object) -> tuple[int, object]:
        game_number = blue_row.get("game") if "game" in blue_row else None
        if game_number is None or pd.isna(game_number):
            game_number = red_row.get("game") if "game" in red_row else None
        if game_number is None or pd.isna(game_number):
            return (1, str(gameid))
        try:
            return (0, int(game_number))
        except (TypeError, ValueError):
            return (0, str(game_number))

    def _compute_game_draft_stats(self, blue_row: pd.Series, red_row: pd.Series) -> dict[str, np.ndarray]:
        size = self.num_champions - 1
        blue_bans = np.zeros(size, dtype=np.float32)
        red_bans = np.zeros(size, dtype=np.float32)
        blue_pick_counts = np.zeros(size, dtype=np.float32)
        red_pick_counts = np.zeros(size, dtype=np.float32)
        blue_availability_counts = np.zeros(size, dtype=np.float32)
        red_availability_counts = np.zeros(size, dtype=np.float32)
        available_mask = np.ones(size, dtype=np.float32)

        for side, action_type, action_number in DRAFT_ORDER:
            row = blue_row if side == "blue" else red_row
            column_prefix = "ban" if action_type == "ban" else "pick"
            column_name = f"{column_prefix}{action_number}"
            champion_name = row.get(column_name)
            champion_index = self._normalize_champion_id(champion_name)

            if action_type == "pick":
                if side == "blue":
                    blue_availability_counts += available_mask
                    if champion_index > 0:
                        blue_pick_counts[champion_index - 1] += 1.0
                else:
                    red_availability_counts += available_mask
                    if champion_index > 0:
                        red_pick_counts[champion_index - 1] += 1.0
            else:
                if champion_index > 0:
                    if side == "blue":
                        blue_bans[champion_index - 1] = 1.0
                    else:
                        red_bans[champion_index - 1] = 1.0

            if champion_index > 0:
                available_mask[champion_index - 1] = 0.0

        return {
            "blue_bans": blue_bans,
            "red_bans": red_bans,
            "blue_pick_counts": blue_pick_counts,
            "red_pick_counts": red_pick_counts,
            "blue_availability_counts": blue_availability_counts,
            "red_availability_counts": red_availability_counts,
        }

    def _build_game_contexts(self) -> list[dict[str, object]]:
        contexts: list[dict[str, object]] = []
        grouped_games = self.data.groupby(["seriesid", "gameid"])

        for (seriesid, gameid), game_rows in grouped_games:
            blue_rows = game_rows[game_rows["side"].str.lower() == "blue"]
            red_rows = game_rows[game_rows["side"].str.lower() == "red"]

            if blue_rows.empty or red_rows.empty:
                continue

            blue_row = blue_rows.iloc[0]
            red_row = red_rows.iloc[0]
            game_date = self._parse_date_value(blue_row.get("date")) or self._parse_date_value(
                red_row.get("date")
            )
            stats = self._compute_game_draft_stats(blue_row, red_row)
            contexts.append(
                {
                    "seriesid": seriesid,
                    "gameid": gameid,
                    "game_sort_key": self._game_sort_key(blue_row, red_row, gameid),
                    "date": game_date,
                    "blue_team": normalize_category(blue_row.get("teamid")),
                    "red_team": normalize_category(red_row.get("teamid")),
                    **stats,
                }
            )

        return contexts

    def _build_series_ban_priors(self, contexts: list[dict[str, object]]) -> dict[tuple[str, object], np.ndarray]:
        series_groups: dict[str, list[dict[str, object]]] = {}
        for context in contexts:
            seriesid = context.get("seriesid")
            if seriesid is None:
                continue
            series_groups.setdefault(str(seriesid), []).append(context)

        series_priors: dict[tuple[str, object], np.ndarray] = {}
        size = self.num_champions - 1

        for seriesid, games in series_groups.items():
            games.sort(key=lambda entry: entry["game_sort_key"])
            blue_counts = np.zeros(size, dtype=np.float32)
            red_counts = np.zeros(size, dtype=np.float32)
            games_seen = 0

            for game in games:
                if games_seen <= 0:
                    series_prior = np.zeros(size, dtype=np.float32)
                else:
                    same_side = blue_counts / float(games_seen)
                    opposite_side = red_counts / float(games_seen)
                    overall = (blue_counts + red_counts) / float(games_seen)
                    miss_rate = 1.0 - overall
                    series_prior = (
                        SERIES_PRIOR_SAME_SIDE_WEIGHT * same_side
                        + SERIES_PRIOR_OVERALL_WEIGHT * overall
                        + SERIES_PRIOR_OPPOSITE_WEIGHT * opposite_side
                        - SERIES_PRIOR_MISS_WEIGHT * miss_rate
                    )
                    series_prior = np.clip(series_prior, 0.0, None)
                    series_prior = self._normalize_prior(series_prior)

                series_priors[(seriesid, game["gameid"])] = series_prior
                blue_counts += game["blue_bans"]
                red_counts += game["red_bans"]
                games_seen += 1

        return series_priors

    def _build_team_priors(self, contexts: list[dict[str, object]]) -> dict[tuple[str, object, str], np.ndarray]:
        records_by_team: dict[str, list[dict[str, object]]] = {}
        priors_by_game: dict[tuple[str, object, str], np.ndarray] = {}
        size = self.num_champions - 1
        empty_prior = np.zeros(size, dtype=np.float32)

        for context in contexts:
            seriesid = context.get("seriesid")
            gameid = context.get("gameid")
            game_date = context.get("date")

            for side in ("blue", "red"):
                team_key = context.get(f"{side}_team")
                if not team_key:
                    continue
                if game_date is None or pd.isna(game_date):
                    priors_by_game[(str(seriesid), gameid, team_key)] = empty_prior
                    continue

                if side == "blue":
                    ban_against = np.zeros(size, dtype=np.float32)
                    pick_counts = context["blue_pick_counts"]
                    availability_counts = context["blue_availability_counts"]
                else:
                    ban_against = context["blue_bans"]
                    pick_counts = context["red_pick_counts"]
                    availability_counts = context["red_availability_counts"]

                records_by_team.setdefault(team_key, []).append(
                    {
                        "seriesid": str(seriesid),
                        "gameid": gameid,
                        "date": game_date,
                        "pick_counts": pick_counts,
                        "availability_counts": availability_counts,
                        "ban_against_counts": ban_against,
                    }
                )

        window = pd.Timedelta(days=self.team_priors_window_days)
        for team_key, records in records_by_team.items():
            records.sort(key=lambda entry: entry["date"])
            window_pick = np.zeros(size, dtype=np.float32)
            window_available = np.zeros(size, dtype=np.float32)
            window_bans = np.zeros(size, dtype=np.float32)
            start_idx = 0

            for idx, record in enumerate(records):
                cutoff = record["date"] - window
                while start_idx < idx and records[start_idx]["date"] < cutoff:
                    window_pick -= records[start_idx]["pick_counts"]
                    window_available -= records[start_idx]["availability_counts"]
                    window_bans -= records[start_idx]["ban_against_counts"]
                    start_idx += 1

                pick_prior = self._build_pick_prior(window_pick, window_available)
                ban_prior = self._normalize_prior(window_bans)
                team_prior = self._blend_priors(
                    [
                        (ban_prior, TEAM_PRIOR_BAN_WEIGHT),
                        (pick_prior, TEAM_PRIOR_PICK_WEIGHT),
                    ]
                )

                priors_by_game[(record["seriesid"], record["gameid"], team_key)] = team_prior

                window_pick += record["pick_counts"]
                window_available += record["availability_counts"]
                window_bans += record["ban_against_counts"]

        return priors_by_game

    def _build_early_blue_ban_priors(self) -> dict[tuple[str, object], torch.Tensor]:
        contexts = self._build_game_contexts()
        series_priors = self._build_series_ban_priors(contexts)
        team_priors = self._build_team_priors(contexts)
        priors_by_game: dict[tuple[str, object], torch.Tensor] = {}

        for context in contexts:
            seriesid = str(context.get("seriesid"))
            gameid = context.get("gameid")
            red_team = context.get("red_team")

            series_prior = series_priors.get((seriesid, gameid))
            team_prior = None
            if red_team:
                team_prior = team_priors.get((seriesid, gameid, red_team))

            combined = self._blend_priors(
                [
                    (series_prior, EARLY_BLUE_BAN_PRIOR_SERIES_WEIGHT),
                    (team_prior, EARLY_BLUE_BAN_PRIOR_TEAM_WEIGHT),
                ]
            )
            if combined.sum() <= 0:
                continue

            priors_by_game[(seriesid, gameid)] = torch.tensor(
                combined * self.early_blue_ban_priors_strength,
                dtype=torch.float32,
            )

        return priors_by_game

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
            patch_value = self._patch_value_for_rows(blue_row, red_row)
            game_date = self._parse_date_value(blue_row.get("date")) or self._parse_date_value(
                red_row.get("date")
            )
            game_date_value = int(game_date.value) if game_date is not None else None
            league_key = normalize_category(blue_row.get("league")) or normalize_category(
                red_row.get("league")
            )
            time_bucket = self._time_bucket_for_patch_date(patch_value, game_date)
            priors_key = self._priors_key(patch_value, time_bucket)
            role_priors = None
            role_tally = None
            if self.role_priors_by_patch is not None:
                role_priors = (
                    self.role_priors_by_patch.get(patch_value, self.default_role_priors)
                    if patch_value
                    else self.default_role_priors
                )
                role_tally = {
                    "blue": np.zeros(len(DEFAULT_ROLE_ORDER), dtype=np.float32),
                    "red": np.zeros(len(DEFAULT_ROLE_ORDER), dtype=np.float32),
                }

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

                is_early_blue_ban = (
                    side == "blue" and action_type == "ban" and action_number <= EARLY_BLUE_BAN_MAX
                )
                samples.append(
                    {
                        "draft_sequence": draft_sequence.copy(),
                        "target": champion_index,
                        "already_picked_or_banned": set(used_champions),
                        "seriesid": seriesid,
                        "gameid": gameid,
                        "patch_index": patch_index,
                        "patch": patch_value,
                        "priors_key": priors_key,
                        "action_type": action_type,
                        "side": side,
                        "event_index": event_index,
                        "league_key": league_key,
                        "game_date_value": game_date_value,
                        "league_index": self._league_index(row.get("league")),
                        "team_index": self._team_index(row.get("teamid")),
                        "is_early_blue_ban": is_early_blue_ban,
                    }
                )

                if role_priors is not None and role_tally is not None:
                    if action_type == "pick":
                        role_need = 1.0 - np.minimum(role_tally[side], 1.0)
                        role_bias = role_priors @ role_need
                    else:
                        role_bias = np.zeros(self.num_champions - 1, dtype=np.float32)
                    samples[-1]["role_priors"] = role_bias * self.role_priors_strength

                if sanitized_name:
                    used_champions.add(sanitized_name)

                draft_sequence[event_index] = champion_index

                if role_priors is not None and role_tally is not None and action_type == "pick":
                    role_tally[side] += role_priors[champion_index - 1]

        return samples
