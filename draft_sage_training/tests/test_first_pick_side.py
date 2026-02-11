import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from draft_sage_training.dataset import DraftDataset, resolve_first_pick_side
from draft_sage_training.utils.draft_order import draft_order_for_first_pick


class FirstPickSideTests(unittest.TestCase):
    def test_resolve_first_pick_side_defaults_to_blue(self) -> None:
        blue_row = pd.Series({"side": "Blue", "teamid": 10})
        red_row = pd.Series({"side": "Red", "teamid": 20})
        self.assertEqual("blue", resolve_first_pick_side(blue_row, red_row))

    def test_resolve_first_pick_side_from_explicit_red_value(self) -> None:
        blue_row = pd.Series({"side": "Blue", "teamid": 10, "firstpick": "Red"})
        red_row = pd.Series({"side": "Red", "teamid": 20, "firstpick": "Red"})
        self.assertEqual("red", resolve_first_pick_side(blue_row, red_row))

    def test_resolve_first_pick_side_from_row_flags(self) -> None:
        blue_row = pd.Series({"side": "Blue", "teamid": 10, "firstpick": 0})
        red_row = pd.Series({"side": "Red", "teamid": 20, "firstpick": 1})
        self.assertEqual("red", resolve_first_pick_side(blue_row, red_row))

    def test_red_first_pick_swaps_pick_phase_sides(self) -> None:
        red_first_order = draft_order_for_first_pick("red")
        self.assertEqual(("red", "pick", 1), red_first_order[6])
        self.assertEqual(("blue", "pick", 1), red_first_order[7])
        self.assertEqual(("blue", "pick", 2), red_first_order[8])
        self.assertEqual(("red", "pick", 2), red_first_order[9])
        self.assertEqual(("red", "pick", 3), red_first_order[10])

    def test_dataset_uses_red_first_pick_order_when_available(self) -> None:
        teams_df = pd.DataFrame(
            [
                {
                    "gameid": "1001",
                    "league": "LCS",
                    "split": "Spring",
                    "year": 2026,
                    "date": "2026-02-11 10:00:00",
                    "game": 1,
                    "patch": "15.2",
                    "participantid": 100,
                    "side": "Blue",
                    "teamid": "10",
                    "ban1": "blue_ban_1",
                    "ban2": "blue_ban_2",
                    "ban3": "blue_ban_3",
                    "ban4": "blue_ban_4",
                    "ban5": "blue_ban_5",
                    "pick1": "blue_pick_1",
                    "pick2": "blue_pick_2",
                    "pick3": "blue_pick_3",
                    "pick4": "blue_pick_4",
                    "pick5": "blue_pick_5",
                    "firstpick": "Red",
                },
                {
                    "gameid": "1001",
                    "league": "LCS",
                    "split": "Spring",
                    "year": 2026,
                    "date": "2026-02-11 10:00:00",
                    "game": 1,
                    "patch": "15.2",
                    "participantid": 200,
                    "side": "Red",
                    "teamid": "20",
                    "ban1": "red_ban_1",
                    "ban2": "red_ban_2",
                    "ban3": "red_ban_3",
                    "ban4": "red_ban_4",
                    "ban5": "red_ban_5",
                    "pick1": "red_pick_1",
                    "pick2": "red_pick_2",
                    "pick3": "red_pick_3",
                    "pick4": "red_pick_4",
                    "pick5": "red_pick_5",
                    "firstpick": "Red",
                },
            ]
        )

        champion_names = [
            "blue_ban_1",
            "blue_ban_2",
            "blue_ban_3",
            "blue_ban_4",
            "blue_ban_5",
            "blue_pick_1",
            "blue_pick_2",
            "blue_pick_3",
            "blue_pick_4",
            "blue_pick_5",
            "red_ban_1",
            "red_ban_2",
            "red_ban_3",
            "red_ban_4",
            "red_ban_5",
            "red_pick_1",
            "red_pick_2",
            "red_pick_3",
            "red_pick_4",
            "red_pick_5",
        ]

        mapping_payload = [{"normalized_name": name} for name in champion_names]
        with TemporaryDirectory() as tmp_dir:
            mapping_path = Path(tmp_dir) / "mapping.json"
            mapping_path.write_text(json.dumps(mapping_payload), encoding="utf-8")

            dataset = DraftDataset(teams_df=teams_df, champion_mapping_path=str(mapping_path))

        self.assertEqual(20, len(dataset.samples))
        self.assertEqual("red", dataset.samples[0]["side"])
        self.assertEqual(
            dataset._normalize_champion_id("red_ban_1"),
            dataset.samples[0]["target"],
        )
        self.assertEqual("red", dataset.samples[6]["side"])
        self.assertEqual(
            dataset._normalize_champion_id("red_pick_1"),
            dataset.samples[6]["target"],
        )
        self.assertEqual("blue", dataset.samples[8]["side"])
        self.assertEqual(
            dataset._normalize_champion_id("blue_pick_2"),
            dataset.samples[8]["target"],
        )


if __name__ == "__main__":
    unittest.main()
