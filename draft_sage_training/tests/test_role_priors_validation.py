import unittest

from draft_sage_training.utils.role_priors import (
    DEFAULT_ROLE_ORDER,
    validate_role_priors_payload,
)


class RolePriorsValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.champions = ["aatrox", "ahri"]
        self.base_payload = {
            "schema_version": "1.0",
            "weight_type": "role-priors",
            "patch": "15.1",
            "normalization": "sum-to-1",
            "roles": DEFAULT_ROLE_ORDER,
            "weights": {
                "aatrox": {"top": 1.0, "jungle": 0.0, "mid": 0.0, "bot": 0.0, "support": 0.0},
                "ahri": {"top": 0.0, "jungle": 0.0, "mid": 1.0, "bot": 0.0, "support": 0.0},
            },
        }

    def test_valid_payload_passes(self) -> None:
        validate_role_priors_payload(self.base_payload, self.champions)

    def test_unknown_champion_rejected(self) -> None:
        payload = dict(self.base_payload)
        payload["weights"] = dict(self.base_payload["weights"])
        payload["weights"]["unknown"] = dict(payload["weights"]["ahri"])
        with self.assertRaises(ValueError):
            validate_role_priors_payload(payload, self.champions)

    def test_missing_champion_rejected(self) -> None:
        payload = dict(self.base_payload)
        payload["weights"] = {"aatrox": self.base_payload["weights"]["aatrox"]}
        with self.assertRaises(ValueError):
            validate_role_priors_payload(payload, self.champions)

    def test_negative_weight_rejected(self) -> None:
        payload = dict(self.base_payload)
        payload["weights"] = dict(self.base_payload["weights"])
        payload["weights"]["ahri"] = dict(payload["weights"]["ahri"])
        payload["weights"]["ahri"]["mid"] = -0.1
        payload["weights"]["ahri"]["support"] = 1.1
        with self.assertRaises(ValueError):
            validate_role_priors_payload(payload, self.champions)

    def test_role_keys_must_match(self) -> None:
        payload = dict(self.base_payload)
        payload["weights"] = dict(self.base_payload["weights"])
        payload["weights"]["ahri"] = {"mid": 1.0}
        with self.assertRaises(ValueError):
            validate_role_priors_payload(payload, self.champions)


if __name__ == "__main__":
    unittest.main()
