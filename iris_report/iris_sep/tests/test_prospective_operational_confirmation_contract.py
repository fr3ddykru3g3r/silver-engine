import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

IRIS_ROOT = Path(__file__).resolve().parents[1]
RUNNER = IRIS_ROOT / "tools" / "run_custodian_prospective_episode_evaluation_v1.py"
CONFIG = IRIS_ROOT / "config" / "prospective_operational_confirmation_v1_preregistration_2026-09-11.json"
spec = importlib.util.spec_from_file_location("custodian_eval", RUNNER)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


class ProspectiveContractTests(unittest.TestCase):
    def test_preregistration_protects_pool_and_information_floor(self):
        cfg = json.loads(CONFIG.read_text())
        self.assertEqual(cfg["study_id"], m.STUDY_ID)
        self.assertEqual(cfg["protected_pool"]["not_before_utc"], "2025-09-10T00:00:00Z")
        self.assertEqual(cfg["protected_pool"]["development_side_outcome_access"], "FORBIDDEN")
        self.assertEqual(cfg["information_floor"]["minimum_distinct_onset_episodes"], 50)
        self.assertEqual(cfg["information_floor"]["minimum_quiet_blocks"], 500)
        self.assertEqual(cfg["bootstrap"]["replicates"], 10000)
        self.assertEqual(cfg["bootstrap"]["seed"], 20260911)

    def synthetic_rows(self):
        rows = []
        for model, alerts in [
            ("past_proton_active_proxy", [0, 1, 0, 0]),
            ("joint_operational_model", [1, 1, 0, 0]),
        ]:
            base = [
                ("2025-09-11T00:00:00Z", alerts[0], .7, 1, 1, 1, 0, "epA", "", 0),
                ("2025-09-12T00:00:00Z", alerts[1], .8, 1, 0, 0, 1, "epA", "", 0),
                ("2025-09-15T00:00:00Z", alerts[2], .2, 0, 1, 0, 0, "", "qb1", 0),
                ("2025-09-22T00:00:00Z", alerts[3], .1, 0, 1, 0, 0, "", "qb2", 0),
            ]
            for issue, alert, probability, mapped, eligible, onset, active, episode, block, ambiguous in base:
                rows.append(m.Row(
                    m.parse_utc(issue), m.parse_utc(issue), model, alert, probability,
                    mapped, eligible, onset, active, episode, block, ambiguous,
                ))
        return rows

    def test_episode_normalization_and_persistence_exclusion(self):
        rows = self.synthetic_rows()
        model = "joint_operational_model"
        mult = m.episode_multiplicities(rows, model)
        self.assertEqual(mult["epA"], 2)
        vals = []
        model_rows = [r for r in rows if r.model == model]
        m._CURRENT_MODEL_ROWS = model_rows
        for row in model_rows:
            wl = m.row_weight_and_label(row, "EPISODE_NORMALIZED_OCCURRENCE", mult)
            if wl and wl[1] == 1:
                vals.append(wl[0])
        self.assertAlmostEqual(sum(vals), 1.0)
        onset = m.metrics(rows, model, "NEW_ONSET_CAUSAL")
        self.assertEqual(onset["TP"], 1.0)
        self.assertEqual(onset["FP"], 0.0)

    def test_shared_unit_bootstrap_is_deterministic(self):
        rows = self.synthetic_rows()
        a = m.bootstrap_contrast(rows, "joint_operational_model", "NEW_ONSET_CAUSAL", "MAPPED_OCCURRENCE", 7, 50)
        b = m.bootstrap_contrast(rows, "joint_operational_model", "NEW_ONSET_CAUSAL", "MAPPED_OCCURRENCE", 7, 50)
        self.assertEqual(a, b)
        self.assertEqual(a["n_episode_units"], 1)
        self.assertEqual(a["n_quiet_blocks"], 2)

    def test_manifest_requires_causal_feature_verification(self):
        base = json.loads(CONFIG.read_text())
        base["status"] = "EXECUTION_FROZEN"
        base["frozen_models"] = {
            "past_proton_active_proxy": {
                "artifact_or_rule_sha256": "a" * 64,
                "feature_schema_sha256": "b" * 64,
                "features_verified_causal": False,
            }
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            path.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "unverified feature"):
                m.load_manifest(path)

    def test_prediction_after_issue_is_rejected(self):
        models = {"past_proton_active_proxy": {}}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sealed.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=sorted(m.REQUIRED_COLUMNS))
                writer.writeheader()
                writer.writerow({
                    "issue_time_utc": "2025-09-11T00:00:00Z",
                    "prediction_timestamp_utc": "2025-09-11T00:00:01Z",
                    "model_id": "past_proton_active_proxy",
                    "alert": "0",
                    "probability": "0.1",
                    "mapped_occurrence": "0",
                    "onset_eligible": "1",
                    "onset_label": "0",
                    "active_at_issue": "0",
                    "episode_id": "",
                    "quiet_block_id": "q",
                    "ambiguous_positive": "0",
                })
            with self.assertRaisesRegex(ValueError, "after issue_time"):
                m.load_rows(path, models)


if __name__ == "__main__":
    unittest.main()
