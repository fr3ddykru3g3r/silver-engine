from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_episode_normalized_development_benchmark_v1_compat as benchmark_compat
import tools.run_onset_forecaster_phase2_v1 as phase2
import tools.run_onset_forecaster_phase2_v1_execution as execution


ROOT = Path(__file__).resolve().parents[1]


def test_phase2_contract_keeps_protected_outcomes_sealed() -> None:
    cfg = json.loads((ROOT / "config" / "onset_forecaster_phase2_v1_preregistration_2026-09-11.json").read_text())
    assert cfg["status"] == "FROZEN_BEFORE_PHASE2_SCORE_INSPECTION"
    assert cfg["separation"]["frozen_benchmark_branch_untouched"] is True
    assert cfg["separation"]["protected_post_2025_outcomes_may_be_accessed"] is False
    assert cfg["data"]["already_active_rows_excluded"] is True


def test_onset_rows_excludes_noneligible_states() -> None:
    frame = pd.DataFrame(
        {
            "role": ["fit", "fit", "fit", "fit"],
            "eligibility_code": [
                "ELIGIBLE_ONSET_POSITIVE",
                "ELIGIBLE_ONSET_NEGATIVE",
                "ALREADY_ACTIVE_PERSISTENCE",
                "UNRESOLVED_GAP",
            ],
        }
    )
    out = phase2.onset_rows(frame, "fit")
    assert out["y_onset"].tolist() == [1, 0]
    assert len(out) == 2


def test_class_ratio_uses_fit_labels_only() -> None:
    assert phase2.class_ratio(np.array([1, 0, 0, 0])) == 3.0


def test_threshold_selector_prefers_lower_far_within_tss_tolerance() -> None:
    y = np.array([1, 1, 0, 0, 0, 0], dtype=int)
    p = np.array([0.9, 0.55, 0.54, 0.20, 0.10, 0.05], dtype=float)
    threshold, diag = phase2.select_threshold(y, p, tolerance=0.20)
    chosen = diag["chosen"]
    assert np.isfinite(threshold)
    assert chosen["TSS"] >= diag["max_tss"] - 0.20 - 1e-12
    near = []
    for th in np.unique(p):
        met = phase2.full_metrics(y, p, float(th))
        if met["TSS"] >= diag["max_tss"] - 0.20 - 1e-12:
            near.append(met)
    assert chosen["FAR"] == min(row["FAR"] for row in near)


def test_execution_adapter_resolves_scientific_count_alias() -> None:
    n = 3
    frame = pd.DataFrame(
        {
            "p_last": [1.0, 2.0, 3.0], "p_max": [2.0, 3.0, 4.0], "p_mean": [1.0, 2.0, 2.0],
            "p_median": [1.0, 2.0, 2.0], "p_count_ge_5": [0.0, 0.0, 0.0], "p_count_ge_8": [0.0, 0.0, 0.0],
            "p_n": [288.0] * n, "p_max_pos_log_change": [0.1, 0.2, 0.3],
            "a_last": [1e-7, 2e-7, 3e-7], "a_mean": [1e-7, 2e-7, 2e-7], "a_max": [2e-7, 3e-7, 4e-7],
            "b_last": [1e-6, 2e-6, 3e-6], "b_mean": [1e-6, 2e-6, 2e-6], "b_max": [2e-6, 3e-6, 4e-6],
            "b_n": [1440.0] * n, "b_count_ge_1e-05": [0.0, 1.0, 2.0], "b_count_ge_1e-04": [0.0, 0.0, 1.0],
            "b_max_pos_log_change": [0.2, 0.1, 0.4],
        }
    )
    out, names = execution.corrected_engineered_features(frame)
    assert "eng_b_ge1e4_fraction" in names
    assert out["eng_b_ge1e4_fraction"].notna().all()
    assert len(names) == 15


def test_cached_timezone_feature_adapter_matches_frozen_compatibility_helper() -> None:
    issue = pd.Timestamp("2017-01-02T00:00:00Z")
    proton_times = pd.date_range(issue - pd.Timedelta(hours=24), issue, freq="5min", inclusive="left")
    proton = pd.DataFrame({"time": proton_times, "proton": np.linspace(0.2, 2.0, len(proton_times))})
    xrs_times = pd.date_range(issue - pd.Timedelta(hours=24), issue, freq="1min", inclusive="left")
    xrs = pd.DataFrame({"time": xrs_times, "A": np.linspace(1e-8, 5e-7, len(xrs_times)), "B": np.linspace(1e-7, 2e-5, len(xrs_times))})

    execution._TIME_NS_CACHE.clear()
    for frame, family in ((proton, "proton"), (xrs, "xrs")):
        expected = benchmark_compat.family_features_compat(frame, issue, 0, family)
        actual = execution.family_features_cached(frame, issue, 0, family)
        assert set(actual) == set(expected)
        for name in expected:
            assert np.isclose(actual[name], expected[name], equal_nan=True)

    assert len(execution._TIME_NS_CACHE) == 2
    first = execution._nanosecond_times(proton)
    second = execution._nanosecond_times(proton)
    assert first is second


def test_runtime_correction_receipt_is_explicit() -> None:
    text = (ROOT / "architecture" / "PHASE2_RUNTIME_COMPATIBILITY_CORRECTION_2026-09-11.md").read_text()
    assert "NO PHASE II MODEL SCORES INSPECTED BEFORE THIS CHANGE" in text
    assert "no modelable rows" in text
    assert "does **not** alter" in text


def test_pre_execution_correction_receipt_is_explicit() -> None:
    text = (ROOT / "architecture" / "PHASE2_PRE_EXECUTION_CORRECTION_2026-09-11.md").read_text()
    assert "RECORDED_BEFORE_PHASE2_REAL_DATA_SCORE_EXECUTION" in text
    assert "does **not** change" in text
    assert "does not access the sealed post-2025 protected outcome pool" in text
