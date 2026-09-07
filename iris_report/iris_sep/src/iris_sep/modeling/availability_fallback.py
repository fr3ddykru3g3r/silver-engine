"""Availability-conditioned specialist fusion for IRIS-SEP.

This module does not reconstruct missing measurements.  It selects an evidence
combiner that was trained in advance for the expert probabilities that remain
available:

    FULL              -> solar + XRS + proton
    NO_XRS            -> solar + proton
    NO_PROTON         -> solar + XRS
    NO_XRS_OR_PROTON  -> solar only

The final state is deliberately labelled as a solar-only fallback rather than a
one-expert learned stack.  All learned multi-expert states use the same
nonnegative PositiveEvidenceStack used by the promoted clean model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .positive_evidence_stack import EvidenceStackConfig, PositiveEvidenceStack


EXPERT_ORDER = ("SOLAR", "XRS", "PROTON")
STATE_EXPERTS = {
    "FULL": ("SOLAR", "XRS", "PROTON"),
    "NO_XRS": ("SOLAR", "PROTON"),
    "NO_PROTON": ("SOLAR", "XRS"),
    "NO_XRS_OR_PROTON": ("SOLAR",),
}


@dataclass(frozen=True)
class AvailabilityPromotionRule:
    """Predeclared gate for exposing an availability fallback as DEGRADED."""

    minimum_affected_tss: float = 0.0
    minimum_affected_pod: float = 0.50
    maximum_whole_score_brier_delta: float = 0.01
    maximum_whole_score_ece_delta: float = 0.02

    def __post_init__(self) -> None:
        if not -1.0 <= self.minimum_affected_tss <= 1.0:
            raise ValueError("minimum_affected_tss must be in [-1,1]")
        if not 0.0 <= self.minimum_affected_pod <= 1.0:
            raise ValueError("minimum_affected_pod must be in [0,1]")
        if self.maximum_whole_score_brier_delta < 0:
            raise ValueError("maximum Brier delta must be nonnegative")
        if self.maximum_whole_score_ece_delta < 0:
            raise ValueError("maximum ECE delta must be nonnegative")


def validate_state(state: str) -> str:
    if state not in STATE_EXPERTS:
        raise ValueError(f"unknown availability state: {state}")
    return state


def evidence_columns(state: str) -> tuple[int, ...]:
    """Return column indices into evidence ordered SOLAR, XRS, PROTON."""
    state = validate_state(state)
    return tuple(EXPERT_ORDER.index(name) for name in STATE_EXPERTS[state])


def select_evidence(evidence, state: str) -> np.ndarray:
    x = np.asarray(evidence, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(EXPERT_ORDER) or x.shape[0] == 0:
        raise ValueError("evidence must have shape [n,3]")
    if not np.isfinite(x).all():
        raise ValueError("evidence must be finite")
    return x[:, evidence_columns(state)]


def fit_availability_stacks(evidence, labels) -> dict[str, PositiveEvidenceStack]:
    """Fit the three multi-expert states; solar-only uses the solar expert itself."""
    x = np.asarray(evidence, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) != len(y):
        raise ValueError("evidence/labels must align with three expert columns")
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("invalid evidence or binary labels")
    out: dict[str, PositiveEvidenceStack] = {}
    for state in ("FULL", "NO_XRS", "NO_PROTON"):
        cols = evidence_columns(state)
        stack = PositiveEvidenceStack(EvidenceStackConfig(expert_count=len(cols)))
        stack.fit(x[:, cols], y)
        out[state] = stack
    return out


def stacked_raw_probability(
    *,
    state: str,
    full_evidence,
    raw_solar_probability,
    stacks: Mapping[str, PositiveEvidenceStack],
    sigmoid,
) -> np.ndarray:
    """Return pre-calibration probability for one pre-trained availability state."""
    state = validate_state(state)
    solar = np.asarray(raw_solar_probability, dtype=np.float64).reshape(-1)
    evidence = np.asarray(full_evidence, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape != (len(solar), 3):
        raise ValueError("full_evidence and solar probability must align")
    if not np.isfinite(solar).all() or ((solar < 0) | (solar > 1)).any():
        raise ValueError("raw solar probability must be finite in [0,1]")
    if state == "NO_XRS_OR_PROTON":
        return solar.copy()
    if state not in stacks:
        raise ValueError(f"missing fitted stack for {state}")
    return np.asarray(sigmoid(stacks[state].decision_function(select_evidence(evidence, state))), dtype=np.float64)


def degraded_candidate_gate(
    *,
    affected_tss: float | None,
    affected_pod: float | None,
    whole_score_brier_delta: float,
    whole_score_ece_delta: float,
    probabilities_finite: bool,
    rule: AvailabilityPromotionRule | None = None,
) -> dict[str, object]:
    """Apply the preregistered DEGRADED-candidate gate.

    Passing this gate never grants NORMAL status.  It only says the fallback may
    be exposed as an explicitly DEGRADED research forecast on the tested state.
    """
    r = rule or AvailabilityPromotionRule()
    checks = {
        "affected_TSS_positive": affected_tss is not None and np.isfinite(affected_tss) and affected_tss > r.minimum_affected_tss,
        "affected_POD_at_least_half": affected_pod is not None and np.isfinite(affected_pod) and affected_pod >= r.minimum_affected_pod,
        "whole_score_Brier_within_margin": np.isfinite(whole_score_brier_delta) and whole_score_brier_delta <= r.maximum_whole_score_brier_delta,
        "whole_score_ECE_within_margin": np.isfinite(whole_score_ece_delta) and whole_score_ece_delta <= r.maximum_whole_score_ece_delta,
        "probabilities_finite": bool(probabilities_finite),
    }
    passed = all(bool(v) for v in checks.values())
    return {
        "passed": passed,
        "permission": "DEGRADED" if passed else "ABSTAIN",
        "normal_allowed": False,
        "checks": checks,
        "rule": {
            "minimum_affected_tss": r.minimum_affected_tss,
            "minimum_affected_pod": r.minimum_affected_pod,
            "maximum_whole_score_brier_delta": r.maximum_whole_score_brier_delta,
            "maximum_whole_score_ece_delta": r.maximum_whole_score_ece_delta,
        },
    }
