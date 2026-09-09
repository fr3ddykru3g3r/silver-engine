"""Issue-time evidence integrity utilities for the IRIS-SEP reliability study.

This module deliberately does not inspect labels when ranking or gating forecasts.
It evaluates whether the evidence presented to a frozen forecaster is available,
authenticated, causally admissible, schema-equivalent, sufficiently fresh, and
quality-valid at issue time.

The numeric evidence_integrity score is an analysis variable, not a learned
probability and not a replacement for the discrete source-contract gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SourceEvidence:
    """Issue-time evidence record for one source required by a forecast path."""

    name: str
    available: bool
    authenticated: bool
    schema_equivalent: bool
    causal: bool
    age_seconds: float
    max_age_seconds: float
    quality_fraction: float = 1.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("source name must be nonempty")
        if not np.isfinite(self.age_seconds) or self.age_seconds < 0:
            raise ValueError(f"{self.name}: age_seconds must be finite and >= 0")
        if not np.isfinite(self.max_age_seconds) or self.max_age_seconds <= 0:
            raise ValueError(f"{self.name}: max_age_seconds must be finite and > 0")
        if not np.isfinite(self.quality_fraction) or not (0.0 <= self.quality_fraction <= 1.0):
            raise ValueError(f"{self.name}: quality_fraction must be in [0, 1]")


@dataclass(frozen=True)
class IntegrityAssessment:
    evidence_integrity: float
    admissible: bool
    blockers: tuple[str, ...]
    per_source_integrity: tuple[tuple[str, float], ...]


def _hard_blockers(source: SourceEvidence) -> list[str]:
    blockers: list[str] = []
    if not source.available:
        blockers.append(f"{source.name}:unavailable")
    if not source.authenticated:
        blockers.append(f"{source.name}:unauthenticated")
    if not source.schema_equivalent:
        blockers.append(f"{source.name}:schema_not_equivalent")
    if not source.causal:
        blockers.append(f"{source.name}:not_issue_time_causal")
    if source.age_seconds >= source.max_age_seconds:
        blockers.append(f"{source.name}:stale")
    if source.quality_fraction <= 0.0:
        blockers.append(f"{source.name}:zero_quality")
    return blockers


def source_integrity(source: SourceEvidence) -> float:
    """Return a deterministic [0, 1] issue-time integrity score.

    Hard contract failures map to zero. Otherwise the score is the weaker of
    (a) remaining normalized freshness margin and (b) declared quality fraction.
    """

    source.validate()
    if _hard_blockers(source):
        return 0.0
    freshness_margin = 1.0 - (source.age_seconds / source.max_age_seconds)
    return float(min(freshness_margin, source.quality_fraction))


def assess_evidence(sources: Iterable[SourceEvidence]) -> IntegrityAssessment:
    """Assess a forecast path using only issue-time source metadata."""

    items = tuple(sources)
    if not items:
        raise ValueError("at least one required source is needed")
    blockers: list[str] = []
    per_source: list[tuple[str, float]] = []
    for source in items:
        source.validate()
        blockers.extend(_hard_blockers(source))
        per_source.append((source.name, source_integrity(source)))
    score = min(value for _, value in per_source)
    return IntegrityAssessment(
        evidence_integrity=float(score),
        admissible=not blockers,
        blockers=tuple(blockers),
        per_source_integrity=tuple(per_source),
    )


def forecast_confidence(probability: float) -> float:
    """Map a binary forecast probability to [0, 1] confidence around 0.5."""

    p = float(probability)
    if not np.isfinite(p) or not (0.0 <= p <= 1.0):
        raise ValueError("probability must be finite and in [0, 1]")
    return float(2.0 * abs(p - 0.5))


def integrity_confidence_discordance(probability: float, evidence_integrity: float) -> float:
    """Positive values indicate confidence exceeding evidential support."""

    integrity = float(evidence_integrity)
    if not np.isfinite(integrity) or not (0.0 <= integrity <= 1.0):
        raise ValueError("evidence_integrity must be finite and in [0, 1]")
    return float(forecast_confidence(probability) - integrity)


def unsafe_confidence(
    probability: float,
    evidence_integrity: float,
    *,
    confidence_floor: float = 0.8,
    integrity_ceiling: float = 0.2,
) -> bool:
    """Flag confident forecasts whose issue-time evidence integrity is low."""

    for value, name in (
        (confidence_floor, "confidence_floor"),
        (integrity_ceiling, "integrity_ceiling"),
    ):
        if not np.isfinite(value) or not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1]")
    return (
        forecast_confidence(probability) >= confidence_floor
        and float(evidence_integrity) <= integrity_ceiling
    )


def top_coverage_indices(scores: Sequence[float], coverage: float) -> np.ndarray:
    """Select the highest issue-time scores at a requested coverage."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("scores must be a nonempty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(coverage) or not (0.0 < coverage <= 1.0):
        raise ValueError("coverage must be in (0, 1]")
    keep = max(1, int(np.ceil(len(values) * float(coverage))))
    order = np.argsort(-values, kind="mergesort")
    return np.sort(order[:keep])


def shared_cluster_bootstrap_draws(
    unit_ids: Sequence[str],
    *,
    replicates: int = 10_000,
    seed: int = 20260909,
) -> tuple[tuple[str, ...], ...]:
    """Generate one shared cluster-draw table for all downstream contrasts."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    units = tuple(dict.fromkeys(str(x) for x in unit_ids))
    if not units:
        raise ValueError("unit_ids must contain at least one unit")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(units), size=(replicates, len(units)))
    return tuple(tuple(units[i] for i in row) for row in indices)
