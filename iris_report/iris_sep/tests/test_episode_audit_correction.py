from __future__ import annotations

import pandas as pd
import pytest

from tools.run_episode_normalized_development_benchmark_v1_auditfix import (
    corrected_full_point_table,
    mapped_point_table,
)


def _row(fold: str, issue: str, probability: float, threshold: float, label: int, *, episode: str | None, code: str, onset_label: float, onset_episode: str | None) -> dict:
    return {
        "fold": fold,
        "issue_timestamp": issue,
        "model": "xgb_joint",
        "probability": probability,
        "threshold": threshold,
        "binary_alert": int(probability >= threshold),
        "standard_occurrence_label": label,
        "eligibility_code": code,
        "onset_label": onset_label,
        "occurrence_episode_id": episode,
        "onset_episode_id": onset_episode,
        "quiet_block_id": "QUIET::Q" + issue[-2:],
        "standard_window_weight": 1.0,
        "episode_normalized_occurrence_weight": 1.0 if episode is not None or label == 0 else 0.0,
        "new_onset_weight": 1.0 if code in {"ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"} else 0.0,
        "episode_normalized_onset_weight": 1.0 if code in {"ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"} else 0.0,
    }


def test_oof_point_metrics_use_each_rows_frozen_alert_not_one_global_threshold() -> None:
    # Fold A threshold 0.8: 0.7 must NOT alert. Fold B threshold 0.2: 0.3 must alert.
    rows = [
        _row("A", "2014-01-01", 0.7, 0.8, 1, episode="E1", code="ELIGIBLE_ONSET_POSITIVE", onset_label=1.0, onset_episode="E1"),
        _row("A", "2014-01-02", 0.1, 0.8, 0, episode=None, code="ELIGIBLE_ONSET_NEGATIVE", onset_label=0.0, onset_episode=None),
        _row("B", "2015-01-01", 0.3, 0.2, 1, episode="E2", code="ELIGIBLE_ONSET_POSITIVE", onset_label=1.0, onset_episode="E2"),
        _row("B", "2015-01-02", 0.4, 0.2, 0, episode=None, code="ELIGIBLE_ONSET_NEGATIVE", onset_label=0.0, onset_episode=None),
    ]
    table = corrected_full_point_table(pd.DataFrame(rows))
    result = table[table.view == "WINDOW_OCCURRENCE_STANDARD"].iloc[0]
    assert result.tp == pytest.approx(1.0)
    assert result.fn == pytest.approx(1.0)
    assert result.fp == pytest.approx(1.0)
    assert result.tn == pytest.approx(1.0)
    assert result.tss == pytest.approx(0.0)
    assert result.threshold_count == 2


def test_mapped_sensitivity_excludes_unmapped_positive_without_hiding_it_from_full_point() -> None:
    rows = [
        _row("A", "2014-01-01", 0.9, 0.5, 1, episode="E1", code="ELIGIBLE_ONSET_POSITIVE", onset_label=1.0, onset_episode="E1"),
        _row("A", "2014-01-02", 0.9, 0.5, 1, episode=None, code="DUPLICATE_OR_AMBIGUOUS", onset_label=float("nan"), onset_episode=None),
        _row("A", "2014-01-03", 0.1, 0.5, 0, episode=None, code="ELIGIBLE_ONSET_NEGATIVE", onset_label=0.0, onset_episode=None),
    ]
    frame = pd.DataFrame(rows)
    full = corrected_full_point_table(frame)
    mapped = mapped_point_table(frame)
    assert int(full[full.view == "WINDOW_OCCURRENCE_STANDARD"].iloc[0].positive_rows) == 2
    assert int(mapped[mapped.view == "MAPPED_STANDARD_OCCURRENCE"].iloc[0].positive_rows) == 1
