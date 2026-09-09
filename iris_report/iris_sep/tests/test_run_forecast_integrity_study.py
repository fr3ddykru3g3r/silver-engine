from __future__ import annotations

import numpy as np
import pandas as pd

from tools.run_forecast_integrity_study import evaluate


def _frame() -> pd.DataFrame:
    rows = []
    for role, dates in (
        ("threshold", pd.date_range("2020-01-01", periods=8, freq="D", tz="UTC")),
        ("score", pd.date_range("2020-02-01", periods=8, freq="D", tz="UTC")),
    ):
        for i, date in enumerate(dates):
            rows.append({
                "issue_time": date.isoformat(),
                "role": role,
                "unit_id": f"{role}-u{i//2}",
                "scenario": "XRS_24H",
                "model": "FROZEN_XGB",
                "label": int(i in (2, 6)),
                "probability": [0.05, 0.95, 0.8, 0.2, 0.9, 0.1, 0.85, 0.15][i],
                "evidence_integrity": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4][i],
                "availability_score": [1, 0, 1, 0, 1, 0, 1, 0][i],
                "decision_threshold": 0.5,
            })
    return pd.DataFrame(rows)


def test_evaluate_freezes_cutoff_on_threshold_role_and_emits_shared_draw_digest() -> None:
    out = evaluate(_frame(), coverages=(0.5, 1.0), bootstrap_replicates=20, seed=7)
    assert out["format"] == "IRIS_FORECAST_INTEGRITY_GATE_V1_RESULT"
    assert out["protected_data_accessed"] is False
    assert len(out["shared_bootstrap_draw_table_sha256"]) == 64
    assert len(out["primary_contrasts"]) == 2


def test_mutating_score_labels_does_not_change_selector_cutoffs_or_realized_coverage() -> None:
    a = _frame()
    b = a.copy()
    b.loc[b["role"] == "score", "label"] = 1 - b.loc[b["role"] == "score", "label"]
    out_a = evaluate(a, coverages=(0.5,), bootstrap_replicates=5, seed=11)
    out_b = evaluate(b, coverages=(0.5,), bootstrap_replicates=5, seed=11)
    key = "XRS_24H::FROZEN_XGB"
    for selector in out_a["selectors"]:
        ma = out_a["results"][key]["selectors"][selector]["0.5"]
        mb = out_b["results"][key]["selectors"][selector]["0.5"]
        assert np.isclose(ma["selector_cutoff"], mb["selector_cutoff"])
        assert np.isclose(ma["realized_coverage"], mb["realized_coverage"])


def test_shared_draw_digest_is_identical_for_same_units_seed_and_replicates() -> None:
    a = evaluate(_frame(), coverages=(0.8,), bootstrap_replicates=10, seed=19)
    b = evaluate(_frame(), coverages=(0.8,), bootstrap_replicates=10, seed=19)
    assert a["shared_bootstrap_draw_table_sha256"] == b["shared_bootstrap_draw_table_sha256"]
