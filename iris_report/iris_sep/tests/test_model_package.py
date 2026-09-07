from __future__ import annotations
import unittest

from iris_report.iris_sep.src.iris_sep.model_package import PACKAGE_FORMAT, STATE_EXPERTS, validate_manifest


def valid_manifest():
    def entries(prefix):
        return [{"seed": s, "path": f"models/{prefix}_{s}.json", "sha256": "a" * 64} for s in [7, 13, 26, 42, 73]]
    return {
        "format": PACKAGE_FORMAT,
        "target": "new_sep_10mev_10pfu_within_24h",
        "seeds": [7, 13, 26, 42, 73],
        "feature_families": {"solar": ["s"], "xrs": ["x"], "proton": ["p"]},
        "model_files": {"solar": entries("s"), "xrs": entries("x"), "proton": entries("p")},
        "states": {
            "FULL": {"experts": ["SOLAR", "XRS", "PROTON"], "calibration_intercept": 0.0, "thresholds": {"MAX_TSS": 0.1, "POD80_MIN_FAR": 0.2}, "stack": {"weights": [1, 1, 1]}},
            "NO_XRS": {"experts": ["SOLAR", "PROTON"], "calibration_intercept": 0.0, "thresholds": {"MAX_TSS": 0.1, "POD80_MIN_FAR": 0.2}, "stack": {"weights": [1, 1]}},
            "NO_PROTON": {"experts": ["SOLAR", "XRS"], "calibration_intercept": 0.0, "thresholds": {"MAX_TSS": 0.1, "POD80_MIN_FAR": 0.2}, "stack": {"weights": [1, 1]}},
            "NO_XRS_OR_PROTON": {"experts": ["SOLAR"], "calibration_intercept": 0.0, "thresholds": {"MAX_TSS": 0.1, "POD80_MIN_FAR": 0.2}, "stack": None}
        }
    }


class ModelPackageTests(unittest.TestCase):
    def test_valid_manifest(self):
        validate_manifest(valid_manifest())

    def test_reject_overlapping_features(self):
        manifest = valid_manifest()
        manifest["feature_families"]["xrs"] = ["s"]
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_reject_path_traversal(self):
        manifest = valid_manifest()
        manifest["model_files"]["solar"][0]["path"] = "../escape.json"
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_reject_missing_availability_state(self):
        manifest = valid_manifest()
        del manifest["states"]["NO_PROTON"]
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_state_contract_is_complete(self):
        self.assertEqual(set(STATE_EXPERTS), {"FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON"})


if __name__ == "__main__":
    unittest.main()
