import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "missingness_recovery_contract_v1.json"


class MissingnessContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text())

    def test_contract_keeps_final_target_and_locked_test_closed(self):
        self.assertEqual(
            self.contract["final_target_unchanged"],
            "new_sep_10mev_10pfu_within_24h",
        )
        self.assertFalse(self.contract["locked_test_access"])
        self.assertEqual(
            self.contract["status"],
            "PREREGISTERED_SOURCE_ONLY_NOT_YET_EXECUTED",
        )

    def test_structural_unavailability_cannot_be_promoted_to_observation(self):
        principles = self.contract["principles"]
        self.assertTrue(principles["reconstruction_is_not_observation"])
        self.assertTrue(principles["structural_unavailability_is_not_imputed_as_observed_data"])
        self.assertTrue(principles["structural_vs_transient_classification_requires_authoritative_source_manifest"])
        structural = self.contract["missingness_classes"]["STRUCTURAL_UNAVAILABLE"]
        self.assertIn("Never treat reconstruction as a recovered observation", structural)

    def test_physics_is_late_arm_and_full_mhd_is_not_automatic(self):
        arms = self.contract["recovery_arms_in_order"]
        physics_index = arms.index("PHYSICS_CONSTRAINED_RECONSTRUCTION_FOR_TRANSIENT_GAPS_ONLY")
        self.assertGreater(physics_index, arms.index("CAUSAL_FORWARD_FILL_WHERE_DEFINED"))
        self.assertGreater(physics_index, arms.index("TRAIN_FIT_MEDIAN_OR_SIMPLE_CAUSAL_STATISTICAL"))
        self.assertFalse(self.contract["full_mhd_policy"]["automatic_inclusion"])
        self.assertIn(
            "source regime authoritatively supports the reconstructed quantity",
            self.contract["full_mhd_policy"]["minimum_requirements"],
        )


if __name__ == "__main__":
    unittest.main()
