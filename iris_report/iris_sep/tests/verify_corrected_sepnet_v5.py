"""Verify the persisted development adapter, including weights and metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools import train_corrected_sepnet_o_v5 as adapter


def verify():
    root = Path(__file__).resolve().parents[1]
    approved_receipts = {
        "faithful": "9583f89130965d85dadb99a7e4384afb9d8a16af853cac22a280fb5040ae7ee6",
        "episode_balanced": "073940a174e1b82d8af92eea94942f5ee69cae168b2a116cb8e0dafd116b8995",
    }
    frame = pd.read_csv(adapter.SOURCE, float_precision="round_trip")
    for suffix, mode in (("faithful", "faithful_row_weighted"), ("episode_balanced", "episode_balanced")):
        directory = root / "artifacts" / f"corrected_sepnet_o_v1_{suffix}_v5"
        assert adapter.sha256_file(directory / "receipt.json") == approved_receipts[suffix]
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["artifact_version"] == "v5"
        assert receipt["experiment_mode"] == mode
        assert receipt["locked_test_accessed"] is False
        assert receipt["headline_eligible_roles"] == []
        assert receipt["source_code_sha256"]["adapter"] == adapter.sha256_file(Path(adapter.__file__))
        for field, filename in {
            "run_config_sha256": "run_config.json",
            "observed_feature_mask_sha256": "observed_feature_mask.npz",
            "training_weights_sha256": "training_weights.npz",
            "general_episode_mapping_sha256": "general_episode_mapping.csv",
            "predictions_sha256": "development_predictions.csv",
            "preprocessing_sha256": "preprocessing.pkl",
            "preprocessing_receipt_sha256": "preprocessing_receipt.json",
        }.items():
            assert receipt[field] == adapter.sha256_file(directory / filename), field
        config = json.loads((directory / "run_config.json").read_text())
        assert config["mode"] == mode and config["version"] == "v5"
        assert config["adapter_sha256"] == receipt["source_code_sha256"]["adapter"]
        assert config["source_sha256"] == adapter.sha256_file(adapter.SOURCE)
        assert config["source_manifest_sha256"] == adapter.sha256_file(adapter.SOURCE_MANIFEST)
        weights, mapping = adapter.episode_weights(frame, mode)
        with np.load(directory / "training_weights.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["weights"], weights)
            np.testing.assert_array_equal(saved["episode_mapping"], mapping)
        features = [c for c in frame if c not in adapter.META]
        with np.load(directory / "observed_feature_mask.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["features"], features)
            np.testing.assert_array_equal(saved["observed"], np.isfinite(frame[features].to_numpy(float)))
        predictions = pd.read_csv(directory / "development_predictions.csv", float_precision="round_trip")
        for column in ("issue_id", "role", "unit_id"):
            np.testing.assert_array_equal(predictions[column], frame[column])
        np.testing.assert_array_equal(predictions.operational_label, frame[adapter.OPERATIONAL])
        threshold_rows = predictions.role.eq("validation_threshold")
        threshold = adapter.select_tss_threshold(
            predictions.loc[threshold_rows, "operational_label"],
            predictions.loc[threshold_rows, "ensemble_probability"], role="validation_threshold")
        assert threshold.threshold == receipt["threshold"]["value"]
        assert threshold.threshold_id == receipt["threshold"]["threshold_id"]
        prevalence = float(predictions.loc[predictions.role.eq("train"), "operational_label"].mean())
        for role in adapter.ROLES:
            rows = predictions[predictions.role.eq(role)]
            metrics = {
                **adapter.probability_metrics(rows.operational_label, rows.ensemble_probability, reference_probability=prevalence),
                **adapter.threshold_metrics(rows.operational_label, rows.ensemble_probability, threshold.threshold),
            }
            for key, value in metrics.items():
                assert np.isclose(value, receipt["metrics_operational_label"][role][key], rtol=0, atol=1e-14), (role, key)
        for seed in receipt["seeds"]:
            for kind in ("best", "last"):
                assert seed[f"{kind}_checkpoint_sha256"] == adapter.sha256_file(directory / seed[f"{kind}_checkpoint"])
    print("CORRECTED_SEPNET_V5_ARTIFACTS_VERIFIED_DEVELOPMENT_ONLY")


if __name__ == "__main__":
    verify()
