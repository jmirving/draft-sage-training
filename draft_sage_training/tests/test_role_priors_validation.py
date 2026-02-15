import unittest

from draft_sage_training.utils.role_priors import (
    DEFAULT_ROLE_ORDER,
    build_causal_patch_weights,
    parse_patch_major_minor,
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


class RolePriorPatchWeightingTests(unittest.TestCase):
    def test_parse_patch_major_minor(self) -> None:
        self.assertEqual(parse_patch_major_minor("26.01"), (26, 1))
        self.assertEqual(parse_patch_major_minor("25.x"), (25, 0))
        self.assertEqual(parse_patch_major_minor("26"), (26, 0))
        self.assertIsNone(parse_patch_major_minor("not-a-patch"))

    def test_causal_patch_weights_upweight_latest_major_and_recency(self) -> None:
        patches = ["25.11", "25.12", "26.01", "26.02", "26.03"]
        weights = build_causal_patch_weights(patches, 4)

        # Same-major patches are strongly favored.
        self.assertGreater(weights["26.01"], weights["25.12"])
        self.assertGreater(weights["26.02"], weights["25.12"])
        self.assertGreater(weights["26.03"], weights["25.12"])

        # Within a major, newer patches get slightly more weight.
        self.assertGreater(weights["26.03"], weights["26.02"])
        self.assertGreater(weights["26.02"], weights["26.01"])

    def test_causal_patch_weights_exclude_future_patches(self) -> None:
        patches = ["25.12", "26.01", "26.02", "26.03"]
        weights_for_2601 = build_causal_patch_weights(patches, 1)
        self.assertEqual(set(weights_for_2601.keys()), {"25.12", "26.01"})


if __name__ == "__main__":
    unittest.main()
