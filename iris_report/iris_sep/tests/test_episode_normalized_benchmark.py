from __future__ import annotations

import numpy as np
import pytest

from tools.episode_normalized_benchmark import (
    EvaluationRow,
    bootstrap_unit_for_row,
    build_shared_bootstrap_draws,
    episode_multiplicity_factor,
    episode_normalized_weights,
    rank_stability,
    score_onset_rows,
)


def test_long_and_short_events_each_total_positive_weight_one() -> None:
    rows = [
        EvaluationRow("a1", "ELIGIBLE_ONSET_POSITIVE", 1, 0.9, episode_id="A"),
        EvaluationRow("a2", "ELIGIBLE_ONSET_POSITIVE", 1, 0.8, episode_id="A"),
        EvaluationRow("a3", "ELIGIBLE_ONSET_POSITIVE", 1, 0.7, episode_id="A"),
        EvaluationRow("b1", "ELIGIBLE_ONSET_POSITIVE", 1, 0.6, episode_id="B"),
    ]
    weights = episode_normalized_weights(rows)
    assert weights[:3].sum() == pytest.approx(1.0)
    assert weights[3] == pytest.approx(1.0)
    assert weights.sum() == pytest.approx(2.0)
    assert episode_multiplicity_factor(rows) == pytest.approx(2.0)


def test_already_active_case_gets_zero_onset_weight() -> None:
    rows = [
        EvaluationRow("p1", "ALREADY_ACTIVE_PERSISTENCE", 1, 0.99, episode_id="A"),
        EvaluationRow("n1", "ELIGIBLE_ONSET_NEGATIVE", 0, 0.1, quiet_block_id="Q1"),
    ]
    weights = episode_normalized_weights(rows)
    assert weights.tolist() == [0.0, 1.0]


def test_episode_normalization_can_change_tss_when_long_event_predictions_differ() -> None:
    rows = [
        EvaluationRow("a1", "ELIGIBLE_ONSET_POSITIVE", 1, 0.9, episode_id="A"),
        EvaluationRow("a2", "ELIGIBLE_ONSET_POSITIVE", 1, 0.9, episode_id="A"),
        EvaluationRow("a3", "ELIGIBLE_ONSET_POSITIVE", 1, 0.9, episode_id="A"),
        EvaluationRow("b1", "ELIGIBLE_ONSET_POSITIVE", 1, 0.1, episode_id="B"),
        EvaluationRow("n1", "ELIGIBLE_ONSET_NEGATIVE", 0, 0.8, quiet_block_id="Q1"),
        EvaluationRow("n2", "ELIGIBLE_ONSET_NEGATIVE", 0, 0.1, quiet_block_id="Q2"),
    ]
    standard = score_onset_rows(rows, 0.5, episode_normalized=False)
    normalized = score_onset_rows(rows, 0.5, episode_normalized=True)
    assert standard["pod"] == pytest.approx(0.75)
    assert normalized["pod"] == pytest.approx(0.5)
    assert standard["tss"] != pytest.approx(normalized["tss"])


def test_null_fixture_has_no_shift_when_each_episode_has_one_window() -> None:
    rows = [
        EvaluationRow("a1", "ELIGIBLE_ONSET_POSITIVE", 1, 0.9, episode_id="A"),
        EvaluationRow("b1", "ELIGIBLE_ONSET_POSITIVE", 1, 0.1, episode_id="B"),
        EvaluationRow("n1", "ELIGIBLE_ONSET_NEGATIVE", 0, 0.8, quiet_block_id="Q1"),
        EvaluationRow("n2", "ELIGIBLE_ONSET_NEGATIVE", 0, 0.1, quiet_block_id="Q2"),
    ]
    standard = score_onset_rows(rows, 0.5, episode_normalized=False)
    normalized = score_onset_rows(rows, 0.5, episode_normalized=True)
    assert standard == normalized
    assert episode_multiplicity_factor(rows) == pytest.approx(1.0)


def test_shared_bootstrap_draw_tensor_is_deterministic_and_reusable() -> None:
    units = ["EPISODE::A", "EPISODE::B", "QUIET::Q1"]
    first = build_shared_bootstrap_draws(units, replicates=20, seed=42)
    second = build_shared_bootstrap_draws(units, replicates=20, seed=42)
    assert np.array_equal(first, second)
    assert first.shape == (20, 3)


def test_bootstrap_unit_keeps_episode_and_quiet_blocks_distinct() -> None:
    positive = EvaluationRow("a1", "ELIGIBLE_ONSET_POSITIVE", 1, 0.8, episode_id="A")
    negative = EvaluationRow("n1", "ELIGIBLE_ONSET_NEGATIVE", 0, 0.2, quiet_block_id="Q7")
    assert bootstrap_unit_for_row(positive) == "EPISODE::A"
    assert bootstrap_unit_for_row(negative) == "QUIET::Q7"


def test_rank_reversal_fixture_is_detected() -> None:
    result = rank_stability(
        {"model_a": 0.60, "model_b": 0.50, "model_c": 0.20},
        {"model_a": 0.40, "model_b": 0.55, "model_c": 0.20},
    )
    assert result["rank_changed"] is True
    assert ("model_a", "model_b") in result["pairwise_reversals"]


def test_model_sets_must_match_for_rank_analysis() -> None:
    with pytest.raises(ValueError, match="model sets differ"):
        rank_stability({"a": 0.5}, {"b": 0.5})


def test_positive_onset_requires_episode_identifier() -> None:
    row = EvaluationRow("bad", "ELIGIBLE_ONSET_POSITIVE", 1, 0.5)
    with pytest.raises(ValueError, match="episode_id"):
        row.validate()


def test_probability_must_be_valid() -> None:
    row = EvaluationRow("bad", "ELIGIBLE_ONSET_NEGATIVE", 0, 1.2, quiet_block_id="Q")
    with pytest.raises(ValueError, match="probability"):
        row.validate()
