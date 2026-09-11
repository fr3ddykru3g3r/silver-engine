from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.run_fpr_controller_v1 import causal_controller, metrics_from_alert


ROOT = Path(__file__).resolve().parents[1]


def test_contract_keeps_protected_outcomes_sealed_and_marks_replay_posthoc() -> None:
    cfg = json.loads((ROOT / "config" / "fpr_controller_v1_preregistration_2026-09-12.json").read_text())
    assert cfg["status"] == "FROZEN_BEFORE_CONTROLLER_REPLAY"
    assert cfg["protected_boundary"]["protected_post_2025_outcomes_may_be_accessed"] is False
    assert cfg["evaluation"]["already_inspected"] is True
    assert cfg["requested_operating_goal"]["maximum_fpr"] == 0.30


def test_controller_cannot_use_unresolved_or_future_labels() -> None:
    times = pd.Series(pd.date_range("2017-01-01", periods=6, freq="D", tz="UTC"))
    y = np.array([0, 0, 0, 0, 0, 0], dtype=int)
    p = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60], dtype=float)
    replay = causal_controller(
        times,
        y,
        p,
        0.05,
        quantile=0.85,
        history_window_days=120,
        minimum_resolved_negative_history=1,
        post_horizon_label_buffer_hours=1,
    )
    # At Jan 2, the Jan 1 24 h label is not yet eligible because of the
    # additional one-hour availability buffer. At Jan 3 it is eligible.
    assert int(replay.loc[1, "resolved_negative_history_size"]) == 0
    assert int(replay.loc[2, "resolved_negative_history_size"]) == 1
    assert replay.loc[2, "controller_threshold"] >= p[0]


def test_threshold_floor_never_relaxes_original_policy() -> None:
    times = pd.Series(pd.date_range("2017-01-01", periods=40, freq="D", tz="UTC"))
    y = np.zeros(40, dtype=int)
    p = np.linspace(0.001, 0.04, 40)
    base = 0.02
    replay = causal_controller(
        times,
        y,
        p,
        base,
        quantile=0.85,
        history_window_days=120,
        minimum_resolved_negative_history=5,
        post_horizon_label_buffer_hours=1,
    )
    assert np.all(replay["controller_threshold"].to_numpy(float) >= base)


def test_metrics_from_alert_matches_expected_confusion() -> None:
    y = np.array([1, 1, 0, 0], dtype=int)
    alert = np.array([1, 0, 1, 0], dtype=int)
    metrics = metrics_from_alert(y, alert)
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["POD"] == 0.5
    assert metrics["FPR"] == 0.5
    assert metrics["TSS"] == 0.0
