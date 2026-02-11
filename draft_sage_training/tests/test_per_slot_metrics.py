import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from draft_sage_training.training import evaluate_model
from draft_sage_training.utils.draft_order import DRAFT_ORDER


class _EvalDataset(Dataset):
    def __init__(self) -> None:
        self.rows = [
            {
                "sample_id": 0,
                "event_index": 0,
                "side": 0,
                "target": 0,
            },
            {
                "sample_id": 1,
                "event_index": 1,
                "side": 1,
                "target": 1,
            },
            {
                "sample_id": 2,
                "event_index": 1,
                "side": 1,
                "target": 2,
            },
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        draft_sequence = torch.zeros(20, dtype=torch.long)
        draft_sequence[0] = row["sample_id"]
        return {
            "draft_sequence": draft_sequence,
            "patch_index": torch.tensor(0, dtype=torch.long),
            "action_type": torch.tensor(0, dtype=torch.long),
            "side": torch.tensor(row["side"], dtype=torch.long),
            "event_index": torch.tensor(row["event_index"], dtype=torch.long),
            "league_index": torch.tensor(0, dtype=torch.long),
            "team_index": torch.tensor(0, dtype=torch.long),
            "output_mask": torch.ones(3, dtype=torch.float32),
            "target": torch.tensor(row["target"], dtype=torch.long),
        }


class _DeterministicModel(torch.nn.Module):
    def forward(self, features: dict) -> torch.Tensor:
        sample_ids = features["draft_sequence"][:, 0]
        batch_size = sample_ids.shape[0]
        outputs = torch.full((batch_size, 3), -5.0, device=sample_ids.device)
        for row_index, sample_id in enumerate(sample_ids.tolist()):
            if sample_id == 0:
                outputs[row_index, 0] = 5.0
            else:
                outputs[row_index, 2] = 5.0
        return outputs


class PerSlotMetricsTests(unittest.TestCase):
    def test_evaluate_model_reports_per_slot_accuracy(self) -> None:
        data_loader = DataLoader(_EvalDataset(), batch_size=3, shuffle=False)
        model = _DeterministicModel()
        loss_function = torch.nn.CrossEntropyLoss()

        _, accuracy, per_slot_accuracy = evaluate_model(
            model=model,
            data_loader=data_loader,
            loss_function=loss_function,
            device="cpu",
        )

        self.assertAlmostEqual(accuracy, 2.0 / 3.0, places=6)
        self.assertEqual(len(per_slot_accuracy), len(DRAFT_ORDER))

        slot1 = per_slot_accuracy[0]
        self.assertEqual(slot1["slot"], 1)
        self.assertEqual(slot1["total"], 1)
        self.assertEqual(slot1["correct"], 1)
        self.assertAlmostEqual(slot1["accuracy"], 1.0, places=6)
        self.assertEqual(slot1["observed_side_counts"]["blue"], 1)
        self.assertEqual(slot1["observed_side_counts"]["red"], 0)

        slot2 = per_slot_accuracy[1]
        self.assertEqual(slot2["slot"], 2)
        self.assertEqual(slot2["total"], 2)
        self.assertEqual(slot2["correct"], 1)
        self.assertAlmostEqual(slot2["accuracy"], 0.5, places=6)
        self.assertEqual(slot2["observed_side_counts"]["blue"], 0)
        self.assertEqual(slot2["observed_side_counts"]["red"], 2)

        slot3 = per_slot_accuracy[2]
        self.assertEqual(slot3["slot"], 3)
        self.assertEqual(slot3["total"], 0)
        self.assertEqual(slot3["correct"], 0)
        self.assertIsNone(slot3["accuracy"])


if __name__ == "__main__":
    unittest.main()
