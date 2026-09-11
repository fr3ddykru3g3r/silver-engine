from __future__ import annotations

import json
from pathlib import Path

from tools.run_fpr_controller_v2 import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_v2_contract_is_explicitly_posthoc_and_sealed() -> None:
    cfg = load_contract()
    assert cfg["status"] == "FROZEN_BEFORE_V2_REPLAY"
    assert cfg["evaluation"]["already_inspected"] is True
    assert cfg["protected_boundary"]["protected_post_2025_outcomes_may_be_accessed"] is False
    assert cfg["controller"]["quantile"] == 0.80
    assert cfg["requested_operating_goal"]["maximum_fpr"] == 0.30
    assert cfg["requested_operating_goal"]["minimum_pod"] == 1.0


def test_v2_does_not_change_source_model_contract() -> None:
    cfg = json.loads((ROOT / "config" / "fpr_controller_v2_development_2026-09-12.json").read_text())
    assert cfg["source_model"] == "direct_onset_engineered"
    assert cfg["protected_boundary"]["source_model_features_hyperparameters_and_probabilities_unchanged"] is True
    assert cfg["controller"]["threshold_floor"] == "original Phase II fixed threshold"
    assert cfg["controller"]["post_horizon_label_buffer_hours"] == 1
