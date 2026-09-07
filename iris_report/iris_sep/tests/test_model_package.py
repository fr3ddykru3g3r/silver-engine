from __future__ import annotations
import copy
import unittest

from iris_report.iris_sep.src.iris_sep.model_package import (
    PACKAGE_FORMAT,
    PACKAGE_FORMAT_V2,
    STATE_EXPERTS,
    STATE_OPERATOR_PERMISSION,
    STATE_STACK_KIND_V3,
    V3_ARCHITECTURE,
    expected_state_feature_schema,
    expected_state_feature_schema_sha256,
    validate_manifest,
)


def _entries(prefix):
    return [
        {"seed": s, "path": f"models/{prefix}_{s}.json", "sha256": "a" * 64}
        for s in [7, 13, 26, 42, 73]
    ]


def valid_manifest():
    return {
        "format": PACKAGE_FORMAT,
        "target": "new_sep_10mev_10pfu_within_24h",
        "seeds": [7, 13, 26, 42, 73],
        "feature_families": {"solar": ["s"], "xrs": ["x"], "proton": ["p"]},
        "model_files": {"solar": _entries("s"), "xrs": _entries("x"), "proton": _entries("p")},
        "states": {
            "FULL": {"experts": ["SOLAR", "XRS", "PROTON"], "calibration_intercept": 0.0, "thresholds": {"MAX_TSS": 0.1, "POD80_MIN_FAR": 0.2}, "stack": {"intercept": 0.0, "weights": [1, 1, 1]}},
            "NO_XRS": {"experts": ["SOLAR", "PROTON"], "calibration_intercept": 0.0, "thresholds": {"MAX_TSS": 0.1, "POD80_MIN_FAR": 0.2}, "stack": {"intercept": 0.0, "weights": [1, 1]}},
            "NO_PROTON": {"experts": ["SOLAR", "XRS"], "calibration_intercept": 0.0, "thresholds": {"MAX_TSS": 0.1, "POD80_MIN_FAR": 0.2}, "stack": {"intercept": 0.0, "weights": [1, 1]}},
            "NO_XRS_OR_PROTON": {"experts": ["SOLAR"], "calibration_intercept": 0.0, "thresholds": {"MAX_TSS": 0.1, "POD80_MIN_FAR": 0.2}, "stack": None}
        }
    }


def valid_v3_manifest():
    manifest = copy.deepcopy(valid_manifest())
    manifest.update({
        "format": PACKAGE_FORMAT_V2,
        "architecture": V3_ARCHITECTURE,
        "fit_prevalence": 0.02,
        "evidence_limit": 6.0,
        "runtime_training_allowed": False,
        "operator_permissions": dict(STATE_OPERATOR_PERMISSION),
        "distillation": {
            "teacher_weight": 0.35,
            "hard_label_weight": 0.65,
            "l2_weight": 0.03,
            "preregistration": "config/availability_distilled_fallback_v3_preregistration_2026-09-07.json",
            "preregistration_sha256": "b" * 64,
        },
    })
    families = manifest["feature_families"]
    manifest["state_feature_schemas"] = {
        state: {
            "schema": expected_state_feature_schema(families, state),
            "sha256": expected_state_feature_schema_sha256(families, state),
        }
        for state in STATE_EXPERTS
    }
    for state, kind in STATE_STACK_KIND_V3.items():
        manifest["states"][state]["stack_kind"] = kind
        if kind == "DISTILLED_POSITIVE_EVIDENCE_STACK":
            manifest["states"][state]["stack"].update({
                "teacher_weight": 0.35,
                "hard_label_weight": 0.65,
                "l2_weight": 0.03,
            })
    return manifest


class ModelPackageTests(unittest.TestCase):
    def test_valid_manifest(self):
        validate_manifest(valid_manifest())

    def test_valid_v3_manifest(self):
        validate_manifest(valid_v3_manifest())

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

    def test_reject_specialist_seed_reordering(self):
        manifest = valid_manifest()
        manifest["model_files"]["solar"][0], manifest["model_files"]["solar"][1] = (
            manifest["model_files"]["solar"][1], manifest["model_files"]["solar"][0]
        )
        with self.assertRaisesRegex(ValueError, "seed order"):
            validate_manifest(manifest)

    def test_reject_missing_availability_state(self):
        manifest = valid_manifest()
        del manifest["states"]["NO_PROTON"]
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_v3_rejects_reordered_state_schema(self):
        manifest = valid_v3_manifest()
        schema = manifest["state_feature_schemas"]["FULL"]["schema"]
        schema["family_order"] = ["XRS", "SOLAR", "PROTON"]
        with self.assertRaises(ValueError):
            validate_manifest(manifest)

    def test_v3_rejects_forged_state_schema_hash(self):
        manifest = valid_v3_manifest()
        manifest["state_feature_schemas"]["NO_XRS"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "schema digest mismatch"):
            validate_manifest(manifest)

    def test_v3_rejects_changed_teacher_weight(self):
        manifest = valid_v3_manifest()
        manifest["distillation"]["teacher_weight"] = 0.50
        with self.assertRaisesRegex(ValueError, "teacher weight"):
            validate_manifest(manifest)

    def test_v3_rejects_permission_upgrade_for_solar_only(self):
        manifest = valid_v3_manifest()
        manifest["operator_permissions"]["NO_XRS_OR_PROTON"] = "DEGRADED"
        with self.assertRaisesRegex(ValueError, "operator permissions"):
            validate_manifest(manifest)

    def test_state_contract_is_complete(self):
        self.assertEqual(set(STATE_EXPERTS), {"FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON"})


if __name__ == "__main__":
    unittest.main()
