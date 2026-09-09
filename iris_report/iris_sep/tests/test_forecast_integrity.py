from __future__ import annotations

import numpy as np

from tools.forecast_integrity import (
    SourceEvidence,
    assess_evidence,
    forecast_confidence,
    integrity_confidence_discordance,
    shared_cluster_bootstrap_draws,
    top_coverage_indices,
    unsafe_confidence,
)


def _good(name: str = "XRS") -> SourceEvidence:
    return SourceEvidence(
        name=name,
        available=True,
        authenticated=True,
        schema_equivalent=True,
        causal=True,
        age_seconds=10.0,
        max_age_seconds=100.0,
        quality_fraction=0.95,
    )


def test_good_sources_are_admissible_and_use_weakest_link() -> None:
    result = assess_evidence([_good("XRS"), _good("PROTON")])
    assert result.admissible is True
    assert result.blockers == ()
    assert np.isclose(result.evidence_integrity, 0.9)


def test_schema_mismatch_hard_blocks_even_high_quality_source() -> None:
    bad = SourceEvidence(
        name="HMI",
        available=True,
        authenticated=True,
        schema_equivalent=False,
        causal=True,
        age_seconds=0.0,
        max_age_seconds=100.0,
        quality_fraction=1.0,
    )
    result = assess_evidence([bad])
    assert result.admissible is False
    assert result.evidence_integrity == 0.0
    assert "HMI:schema_not_equivalent" in result.blockers


def test_stale_or_noncausal_source_hard_blocks() -> None:
    stale = SourceEvidence(
        name="PROTON",
        available=True,
        authenticated=True,
        schema_equivalent=True,
        causal=False,
        age_seconds=100.0,
        max_age_seconds=100.0,
        quality_fraction=1.0,
    )
    result = assess_evidence([stale])
    assert result.admissible is False
    assert result.evidence_integrity == 0.0
    assert "PROTON:not_issue_time_causal" in result.blockers
    assert "PROTON:stale" in result.blockers


def test_high_confidence_can_be_unsupported_by_evidence() -> None:
    assert np.isclose(forecast_confidence(0.95), 0.9)
    assert unsafe_confidence(0.95, 0.0)
    assert np.isclose(integrity_confidence_discordance(0.95, 0.0), 0.9)


def test_selector_is_label_free() -> None:
    score = np.array([0.1, 0.8, 0.5, 0.9])
    labels_a = np.array([0, 0, 1, 1])
    labels_b = 1 - labels_a
    selected_a = top_coverage_indices(score, 0.5)
    selected_b = top_coverage_indices(score, 0.5)
    assert np.array_equal(selected_a, selected_b)
    assert np.array_equal(selected_a, np.array([1, 3]))
    assert not np.array_equal(labels_a, labels_b)


def test_shared_bootstrap_draws_are_reused_and_deterministic() -> None:
    a = shared_cluster_bootstrap_draws(["e1", "e1", "q1", "q2"], replicates=5, seed=7)
    b = shared_cluster_bootstrap_draws(["e1", "e1", "q1", "q2"], replicates=5, seed=7)
    c = shared_cluster_bootstrap_draws(["e1", "e1", "q1", "q2"], replicates=5, seed=8)
    assert a == b
    assert a != c
    assert all(len(draw) == 3 for draw in a)
