from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.episode_semantics import (
    OperationalEpisode,
    attrition_counts,
    build_threshold_episodes,
    classify_window_from_episode_intervals,
    episode_normalized_occurrence_weights,
)


def ep(name: str, start: str, end: str) -> OperationalEpisode:
    return OperationalEpisode(name, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))


def test_persistence_and_onset_are_separated() -> None:
    episodes = [ep("A", "2020-01-02 06:00", "2020-01-03 12:00")]
    persistence = classify_window_from_episode_intervals(
        "2020-01-03 00:00", "2020-01-04 00:00", episodes
    )
    assert persistence.eligibility_code == "ALREADY_ACTIVE_PERSISTENCE"
    assert persistence.interval_occurrence_label == 1

    onset = classify_window_from_episode_intervals(
        "2020-01-02 00:00", "2020-01-03 00:00", episodes
    )
    assert onset.eligibility_code == "ELIGIBLE_ONSET_POSITIVE"
    assert onset.onset_episode_id == "A"


def test_horizon_end_is_exclusive() -> None:
    episodes = [ep("A", "2020-01-03 00:00", "2020-01-03 06:00")]
    row = classify_window_from_episode_intervals(
        "2020-01-02 00:00", "2020-01-03 00:00", episodes
    )
    assert row.eligibility_code == "ELIGIBLE_ONSET_NEGATIVE"
    assert row.interval_occurrence_label == 0


def test_exact_start_at_issue_is_persistence() -> None:
    episodes = [ep("A", "2020-01-02 00:00", "2020-01-03 00:00")]
    row = classify_window_from_episode_intervals(
        "2020-01-02 00:00", "2020-01-03 00:00", episodes
    )
    assert row.eligibility_code == "ALREADY_ACTIVE_PERSISTENCE"


def test_ambiguous_multiple_future_onsets_are_not_silently_assigned() -> None:
    episodes = [
        ep("A", "2020-01-02 06:00", "2020-01-02 08:00"),
        ep("B", "2020-01-02 18:00", "2020-01-02 20:00"),
    ]
    row = classify_window_from_episode_intervals(
        "2020-01-02 00:00", "2020-01-03 00:00", episodes
    )
    assert row.eligibility_code == "DUPLICATE_OR_AMBIGUOUS"
    assert set(row.occurrence_episode_ids) == {"A", "B"}


def test_occurrence_episode_weights_give_each_episode_total_one() -> None:
    ids = ["A", "A", "A", "B", None, None]
    labels = [1, 1, 1, 1, 1, 0]
    w = episode_normalized_occurrence_weights(ids, labels)
    assert w[:3].sum() == pytest.approx(1.0)
    assert w[3] == pytest.approx(1.0)
    assert w[4] == 0.0  # ambiguous/unmapped positive is excluded only here
    assert w[5] == 1.0


def test_threshold_episode_builder_never_bridges_a_gap() -> None:
    times = pd.to_datetime(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 00:20",
            "2020-01-01 00:25",
        ],
        utc=True,
    )
    flux = [12.0, 11.0, 15.0, 5.0]
    episodes = build_threshold_episodes(times, flux, maximum_gap_minutes=5)
    assert len(episodes) == 2
    assert episodes[0].start == times[0]
    assert episodes[1].start == times[2]


def test_threshold_episode_builder_rejects_duplicate_timestamps() -> None:
    times = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:00"], utc=True)
    with pytest.raises(ValueError, match="strictly increasing"):
        build_threshold_episodes(times, [12.0, 13.0])


def test_attrition_counts_reconcile_exactly() -> None:
    codes = ["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE", "ELIGIBLE_ONSET_NEGATIVE"]
    counts = attrition_counts(codes)
    assert counts == {"ELIGIBLE_ONSET_NEGATIVE": 2, "ELIGIBLE_ONSET_POSITIVE": 1}
    assert sum(counts.values()) == len(codes)
