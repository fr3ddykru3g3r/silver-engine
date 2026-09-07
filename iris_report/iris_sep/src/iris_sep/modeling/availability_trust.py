"""Evidence-driven trust promotion for availability-conditioned IRIS-SEP forecasts.

A missing sensor family is not, by itself, evidence that a forecast must be
labelled DEGRADED.  Conversely, a finite probability is not evidence that the
forecast deserves NORMAL trust.

This module separates those two questions.  It allows a pre-trained fallback to
be promoted to NORMAL/VALID only when *independent, frozen* evaluation evidence
shows that the fallback remains acceptably close to the full-data forecast.
Development-only evidence can support DEGRADED or ABSTAIN, but can never promote
NORMAL.

Research decision-support only; this is not operational certification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .availability_fallback import AvailabilityPromotionRule, degraded_candidate_gate, validate_state


TARGET = "new_sep_10mev_10pfu_within_24h"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
NORMAL_ELIGIBLE_STATES = frozenset({"NO_XRS", "NO_PROTON"})
PERMISSIONS = frozenset({"VALID", "DEGRADED", "ABSTAIN"})


@dataclass(frozen=True)
class NormalPromotionRule:
    """Frozen non-inferiority/usefulness requirements for NORMAL trust.

    The numerical margins are intentionally stricter than the existing
    DEGRADED-candidate gate.  They should be frozen before an independent final
    evaluation and must not be retuned after seeing that evaluation.
    """

    maximum_tss_deficit_vs_full: float = 0.05
    maximum_whole_score_brier_delta: float = 0.005
    maximum_whole_score_ece_delta: float = 0.01
    minimum_affected_pod: float = 0.50
    minimum_review_enrichment: float = 5.0
    primary_review_fraction: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum_tss_deficit_vs_full <= 1.0:
            raise ValueError("maximum_tss_deficit_vs_full must be in [0,1]")
        if self.maximum_whole_score_brier_delta < 0:
            raise ValueError("maximum Brier delta must be nonnegative")
        if self.maximum_whole_score_ece_delta < 0:
            raise ValueError("maximum ECE delta must be nonnegative")
        if not 0.0 <= self.minimum_affected_pod <= 1.0:
            raise ValueError("minimum_affected_pod must be in [0,1]")
        if not math.isfinite(self.minimum_review_enrichment) or self.minimum_review_enrichment <= 1.0:
            raise ValueError("minimum_review_enrichment must be finite and >1")
        if not 0.0 < self.primary_review_fraction <= 1.0:
            raise ValueError("primary_review_fraction must be in (0,1]")


@dataclass(frozen=True)
class AvailabilityValidationEvidence:
    """Immutable facts from one availability-state evaluation receipt."""

    availability_state: str
    target: str
    evaluation_scope: str
    evaluation_cohort_sha256: str
    model_package_sha256: str
    locked_configuration_sha256: str
    development_rows_used: bool
    threshold_refit_on_evaluation: bool
    calibration_refit_on_evaluation: bool
    probabilities_finite: bool
    affected_tss: float | None
    affected_pod: float | None
    whole_score_brier_delta_vs_full: float
    whole_score_ece_delta_vs_full: float
    paired_tss_ci_lower_vs_full: float | None
    review_fraction: float
    review_enrichment: float | None

    def __post_init__(self) -> None:
        validate_state(self.availability_state)
        for value in (
            self.evaluation_cohort_sha256,
            self.model_package_sha256,
            self.locked_configuration_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("evaluation bindings must be lowercase SHA-256 values")
        if not isinstance(self.evaluation_scope, str) or not self.evaluation_scope:
            raise ValueError("evaluation_scope is required")
        if not 0.0 < self.review_fraction <= 1.0:
            raise ValueError("review_fraction must be in (0,1]")


def _finite(value: float | None) -> bool:
    return value is not None and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def availability_permission(
    evidence: AvailabilityValidationEvidence,
    *,
    normal_rule: NormalPromotionRule | None = None,
    degraded_rule: AvailabilityPromotionRule | None = None,
) -> dict[str, object]:
    """Return VALID, DEGRADED, or ABSTAIN from frozen evidence only.

    NORMAL/VALID requires an independent locked evaluation, no evaluation-time
    calibration/threshold refit, a fixed 5% review budget, and non-inferiority
    of TSS relative to the full-data model.  If NORMAL fails, the existing
    DEGRADED-candidate gate is evaluated on the same evidence.
    """

    normal = normal_rule or NormalPromotionRule()
    degraded = degraded_rule or AvailabilityPromotionRule()

    independent_checks = {
        "normal_state_eligible": evidence.availability_state in NORMAL_ELIGIBLE_STATES,
        "target_matches": evidence.target == TARGET,
        "independent_locked_scope": evidence.evaluation_scope == "INDEPENDENT_LOCKED_EVALUATION",
        "development_rows_not_used": evidence.development_rows_used is False,
        "threshold_not_refit": evidence.threshold_refit_on_evaluation is False,
        "calibration_not_refit": evidence.calibration_refit_on_evaluation is False,
        "probabilities_finite": evidence.probabilities_finite is True,
        "affected_TSS_positive": _finite(evidence.affected_tss) and float(evidence.affected_tss) > 0.0,
        "affected_POD_sufficient": _finite(evidence.affected_pod) and float(evidence.affected_pod) >= normal.minimum_affected_pod,
        "Brier_noninferior": math.isfinite(evidence.whole_score_brier_delta_vs_full)
        and evidence.whole_score_brier_delta_vs_full <= normal.maximum_whole_score_brier_delta,
        "ECE_noninferior": math.isfinite(evidence.whole_score_ece_delta_vs_full)
        and evidence.whole_score_ece_delta_vs_full <= normal.maximum_whole_score_ece_delta,
        "paired_TSS_noninferior": _finite(evidence.paired_tss_ci_lower_vs_full)
        and float(evidence.paired_tss_ci_lower_vs_full) >= -normal.maximum_tss_deficit_vs_full,
        "review_fraction_frozen": math.isclose(
            evidence.review_fraction, normal.primary_review_fraction, rel_tol=0.0, abs_tol=1e-12
        ),
        "review_enrichment_sufficient": _finite(evidence.review_enrichment)
        and float(evidence.review_enrichment) >= normal.minimum_review_enrichment,
    }
    if all(independent_checks.values()):
        return {
            "permission": "VALID",
            "normal_allowed": True,
            "normal_checks": independent_checks,
            "degraded_gate": None,
        }

    degraded_gate = degraded_candidate_gate(
        affected_tss=evidence.affected_tss,
        affected_pod=evidence.affected_pod,
        whole_score_brier_delta=evidence.whole_score_brier_delta_vs_full,
        whole_score_ece_delta=evidence.whole_score_ece_delta_vs_full,
        probabilities_finite=evidence.probabilities_finite,
        rule=degraded,
    )
    permission = str(degraded_gate["permission"])
    if permission not in PERMISSIONS:
        raise ValueError("invalid degraded gate permission")
    return {
        "permission": permission,
        "normal_allowed": False,
        "normal_checks": independent_checks,
        "degraded_gate": degraded_gate,
    }
