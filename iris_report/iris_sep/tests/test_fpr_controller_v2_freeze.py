from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.run_fpr_controller_v1 import causal_controller
from tools.validate_prospective_fpr_controller_freeze_v1 import (
    load_json,
    validate_contracts,
    validate_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


def run_controller(times: pd.Series, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    return causal_controller(
        times,
        y,
        p,
        0.10,
        quantile=0.80,
        history_window_days=120,
        minimum_resolved_negative_history=3,
        post_horizon_label_buffer_hours=1,
    )


def test_future_labels_and_scores_cannot_change_past_controller_outputs() -> None:
    times = pd.Series(pd.date_range("2030-01-01", periods=20, freq="D", tz="UTC"))
    y = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    p = np.linspace(0.01, 0.95, len(y))
    baseline = run_controller(times, y, p)

    cut = 11
    y_changed = y.copy()
    y_changed[cut:] = 1 - y_changed[cut:]
    p_changed = p.copy()
    p_changed[cut:] = np.linspace(0.99, 0.50, len(y) - cut)
    changed = run_controller(times, y_changed, p_changed)

    np.testing.assert_allclose(
        baseline.loc[: cut - 1, "controller_threshold"].to_numpy(float),
        changed.loc[: cut - 1, "controller_threshold"].to_numpy(float),
    )
    np.testing.assert_array_equal(
        baseline.loc[: cut - 1, "controller_alert"].to_numpy(int),
        changed.loc[: cut - 1, "controller_alert"].to_numpy(int),
    )


def test_current_target_label_cannot_change_current_threshold_or_alert() -> None:
    times = pd.Series(pd.date_range("2030-02-01", periods=12, freq="D", tz="UTC"))
    y = np.zeros(12, dtype=int)
    p = np.array([0.10, 0.12, 0.08, 0.11, 0.14, 0.09, 0.13, 0.15, 0.16, 0.18, 0.17, 0.19])
    baseline = run_controller(times, y, p)

    t = 9
    changed_y = y.copy()
    changed_y[t] = 1
    changed = run_controller(times, changed_y, p)

    assert baseline.loc[t, "controller_threshold"] == changed.loc[t, "controller_threshold"]
    assert baseline.loc[t, "controller_alert"] == changed.loc[t, "controller_alert"]


def test_only_resolved_negative_rows_enter_negative_score_history() -> None:
    times = pd.Series(pd.date_range("2030-03-01", periods=10, freq="D", tz="UTC"))
    # A very high positive score should never raise the negative-score quantile.
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    p = np.array([0.11, 0.12, 0.99, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19])
    replay = run_controller(times, y, p)

    # By the final issue, the positive at index 2 is fully resolved but must
    # still be excluded because controller history is negative-only.
    assert int(replay.loc[9, "resolved_negative_history_size"]) == 7
    assert float(replay.loc[9, "controller_threshold"]) < 0.99


def test_frozen_prospective_contract_matches_v2_policy_and_template_fails_closed() -> None:
    prereg = load_json(ROOT / "config" / "prospective_fpr_controller_confirmation_v1_preregistration_2026-09-12.json")
    freeze = load_json(ROOT / "config" / "fpr_controller_v2_freeze_2026-09-12.json")
    manifest = load_json(ROOT / "config" / "prospective_fpr_controller_execution_manifest_template_v1.json")

    validate_contracts(prereg, freeze)
    result = validate_manifest(manifest)
    assert result["execution_ready"] is False
    assert result["status"] == "BLOCKED_FAIL_CLOSED"
    assert result["protected_outcome_access_performed_by_validator"] is False
    assert "source_model.model_id" in result["missing_required_fields"]


def test_prospective_confirmation_gates_are_not_weaker_than_requested_fpr_target() -> None:
    prereg = json.loads(
        (ROOT / "config" / "prospective_fpr_controller_confirmation_v1_preregistration_2026-09-12.json").read_text()
    )
    gates = prereg["confirmation_gates"]["all_required"]
    assert "controller point FPR <= 0.30" in gates
    assert any("POD" in gate and "-0.10" in gate for gate in gates)
    assert any("TSS" in gate and "lower bound > 0" in gate for gate in gates)
    assert prereg["information_floor"]["minimum_distinct_onset_episodes"] == 50
