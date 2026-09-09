"""Fail-closed missing-data resolution for operator-facing IRIS-SEP research.

This module separates *where a value came from* from *whether a forecast may be
shown normally*.  It never relabels a reconstructed value as an observation.

Resolution order:

    PRIMARY_OBSERVED -> ALTERNATE_OBSERVED -> RECONSTRUCTED ->
    MASK_AWARE_DEGRADED -> ABSTAIN

The implementation is deliberately policy-driven.  A real alternate feed needs
explicit harmonization evidence; a reconstruction needs a causal provenance
record, a validated method/horizon and an evidence receipt.  Structural
historical unavailability cannot enter the reconstruction branch.

This is research decision-support plumbing, not operational certification.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Mapping, Sequence

from .missingness_recovery import ReconstructionProvenance, audit_forecast_time_reconstruction


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MISSINGNESS_CLASSES = frozenset({"OBSERVED", "TRANSIENT_MISSING", "STRUCTURAL_UNAVAILABLE"})
VALUE_ORIGINS = frozenset({"PRIMARY_OBSERVED", "ALTERNATE_OBSERVED", "RECONSTRUCTED", "NONE"})
FORECAST_PERMISSIONS = frozenset({"VALID", "DEGRADED", "ABSTAIN"})


@dataclass(frozen=True)
class ObservedSourceRecord:
    """Forecast-time provenance for one genuinely observed source value."""

    source_id: str
    source_revision: str
    observed_at: datetime
    published_at: datetime
    payload_sha256: str
    harmonization_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_revision:
            raise ValueError("source_id and source_revision are required")
        if self.observed_at.tzinfo is None or self.published_at.tzinfo is None:
            raise ValueError("source timestamps must be timezone-aware")
        if self.observed_at > self.published_at:
            raise ValueError("an observation cannot be published before it is observed")
        if _SHA256.fullmatch(self.payload_sha256) is None:
            raise ValueError("payload_sha256 must be a lowercase SHA-256")
        if self.harmonization_evidence_sha256 is not None and _SHA256.fullmatch(
            self.harmonization_evidence_sha256
        ) is None:
            raise ValueError("harmonization evidence must be a lowercase SHA-256")


@dataclass(frozen=True)
class MissingDataResolutionPolicy:
    """Frozen admission policy for one operator-facing missing-data resolver."""

    policy_id: str
    maximum_observation_age_hours: Mapping[str, float]
    allowed_primary_sources: Mapping[str, tuple[str, ...]]
    allowed_alternate_sources: Mapping[str, tuple[str, ...]]
    allowed_source_revisions: Mapping[str, tuple[str, ...]]
    allowed_reconstruction_method_ids: Mapping[str, tuple[str, ...]]
    maximum_reconstruction_gap_hours: Mapping[str, float]
    maximum_reconstruction_uncertainty: Mapping[str, float]
    reconstruction_evidence_sha256: Mapping[str, str]
    structural_mask_supported_modalities: tuple[str, ...] = ()
    mask_aware_transient_supported_modalities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id is required")
        modalities = set(self.maximum_observation_age_hours)
        required_maps = (
            self.allowed_primary_sources,
            self.allowed_alternate_sources,
            self.allowed_source_revisions,
            self.allowed_reconstruction_method_ids,
            self.maximum_reconstruction_gap_hours,
            self.maximum_reconstruction_uncertainty,
            self.reconstruction_evidence_sha256,
        )
        if not modalities or any(set(mapping) != modalities for mapping in required_maps):
            raise ValueError("all policy mappings must define the same non-empty modality set")
        if not set(self.structural_mask_supported_modalities).issubset(modalities):
            raise ValueError("unknown structural-mask modality")
        if not set(self.mask_aware_transient_supported_modalities).issubset(modalities):
            raise ValueError("unknown mask-aware transient modality")
        for modality in modalities:
            age = self.maximum_observation_age_hours[modality]
            gap = self.maximum_reconstruction_gap_hours[modality]
            uncertainty = self.maximum_reconstruction_uncertainty[modality]
            if not isinstance(age, (int, float)) or isinstance(age, bool) or not math.isfinite(age) or age <= 0:
                raise ValueError("maximum observation ages must be positive and finite")
            if not isinstance(gap, (int, float)) or isinstance(gap, bool) or not math.isfinite(gap) or gap < 0:
                raise ValueError("maximum reconstruction gaps must be finite and nonnegative")
            if (
                not isinstance(uncertainty, (int, float))
                or isinstance(uncertainty, bool)
                or not math.isfinite(uncertainty)
                or not 0 <= uncertainty <= 1
            ):
                raise ValueError("maximum reconstruction uncertainty must be in [0,1]")
            if _SHA256.fullmatch(self.reconstruction_evidence_sha256[modality]) is None:
                raise ValueError("each reconstruction evidence binding must be a lowercase SHA-256")


@dataclass(frozen=True)
class MissingDataResolution:
    modality: str
    missingness_class: str
    value_origin: str
    forecast_permission: str
    selected_source_id: str | None
    selected_source_revision: str | None
    reconstruction_method_id: str | None
    gap_hours: float | None
    normalized_uncertainty: float | None
    reconstruction_is_observation: bool
    reasons: tuple[str, ...]
    evidence_sha256: str | None

    def __post_init__(self) -> None:
        if self.missingness_class not in MISSINGNESS_CLASSES:
            raise ValueError("invalid missingness_class")
        if self.value_origin not in VALUE_ORIGINS:
            raise ValueError("invalid value_origin")
        if self.forecast_permission not in FORECAST_PERMISSIONS:
            raise ValueError("invalid forecast_permission")
        if self.reconstruction_is_observation:
            raise ValueError("a reconstruction may never be labelled as an observation")

    def as_dict(self) -> dict:
        return {
            "modality": self.modality,
            "missingness_class": self.missingness_class,
            "value_origin": self.value_origin,
            "forecast_permission": self.forecast_permission,
            "selected_source_id": self.selected_source_id,
            "selected_source_revision": self.selected_source_revision,
            "reconstruction_method_id": self.reconstruction_method_id,
            "gap_hours": self.gap_hours,
            "normalized_uncertainty": self.normalized_uncertainty,
            "reconstruction_is_observation": False,
            "reasons": list(self.reasons),
            "evidence_sha256": self.evidence_sha256,
        }


def _observed_source_reasons(
    record: ObservedSourceRecord,
    *,
    modality: str,
    issued_at: datetime,
    policy: MissingDataResolutionPolicy,
    alternate: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if record.published_at > issued_at:
        reasons.append("SOURCE_NOT_AVAILABLE_AT_ISSUE_TIME")
    age_hours = (issued_at - record.observed_at).total_seconds() / 3600.0
    if age_hours < 0:
        reasons.append("OBSERVATION_AFTER_ISSUE_TIME")
    elif age_hours > policy.maximum_observation_age_hours[modality]:
        reasons.append("SOURCE_TOO_STALE")
    allowed_ids = (
        policy.allowed_alternate_sources[modality]
        if alternate
        else policy.allowed_primary_sources[modality]
    )
    if record.source_id not in allowed_ids:
        reasons.append("SOURCE_ID_NOT_ALLOWED")
    if record.source_revision not in policy.allowed_source_revisions[modality]:
        reasons.append("SOURCE_REVISION_NOT_ALLOWED")
    if alternate and record.harmonization_evidence_sha256 is None:
        reasons.append("ALTERNATE_SOURCE_HARMONIZATION_UNVERIFIED")
    return tuple(sorted(set(reasons)))


def _best_alternate(
    records: Sequence[ObservedSourceRecord],
    *,
    modality: str,
    issued_at: datetime,
    policy: MissingDataResolutionPolicy,
) -> ObservedSourceRecord | None:
    valid = [
        record
        for record in records
        if not _observed_source_reasons(
            record, modality=modality, issued_at=issued_at, policy=policy, alternate=True
        )
    ]
    if not valid:
        return None
    # Prefer the newest real observation; deterministic source ID breaks exact ties.
    return sorted(valid, key=lambda r: (-r.observed_at.timestamp(), r.source_id))[0]


def resolve_modality_input(
    *,
    modality: str,
    missingness_class: str,
    issued_at: datetime,
    policy: MissingDataResolutionPolicy,
    primary_observation: ObservedSourceRecord | None = None,
    alternate_observations: Sequence[ObservedSourceRecord] = (),
    reconstruction_provenance: ReconstructionProvenance | None = None,
    reconstruction_gap_hours: float | None = None,
    reconstruction_evidence_sha256: str | None = None,
) -> MissingDataResolution:
    """Resolve one modality without conflating recovery with observation.

    A valid primary or alternate observation may produce ``VALID``.  A validated
    reconstruction produces ``DEGRADED`` because the primary measurement was
    unavailable even when the recovered value is scientifically admissible.
    Unsupported criticality is handled by the caller's operator policy; this
    resolver itself fails closed to ``ABSTAIN`` whenever no explicitly admitted
    input path remains.
    """

    if issued_at.tzinfo is None:
        raise ValueError("issued_at must be timezone-aware")
    if modality not in policy.maximum_observation_age_hours:
        raise ValueError("modality is not defined by the resolution policy")
    if missingness_class not in MISSINGNESS_CLASSES:
        raise ValueError("unsupported missingness_class")

    accumulated: list[str] = []

    if primary_observation is not None:
        primary_reasons = _observed_source_reasons(
            primary_observation,
            modality=modality,
            issued_at=issued_at,
            policy=policy,
            alternate=False,
        )
        if not primary_reasons:
            return MissingDataResolution(
                modality, missingness_class, "PRIMARY_OBSERVED", "VALID",
                primary_observation.source_id, primary_observation.source_revision,
                None, None, None, False, (), primary_observation.payload_sha256,
            )
        accumulated.extend(primary_reasons)

    alternate = _best_alternate(
        alternate_observations,
        modality=modality,
        issued_at=issued_at,
        policy=policy,
    )
    if alternate is not None:
        return MissingDataResolution(
            modality, missingness_class, "ALTERNATE_OBSERVED", "VALID",
            alternate.source_id, alternate.source_revision,
            None, None, None, False,
            tuple(sorted(set(accumulated))), alternate.harmonization_evidence_sha256,
        )

    if missingness_class == "STRUCTURAL_UNAVAILABLE":
        if reconstruction_provenance is not None:
            accumulated.append("STRUCTURAL_UNAVAILABILITY_CANNOT_BE_RECONSTRUCTED_AS_OBSERVED")
        if modality in policy.structural_mask_supported_modalities:
            return MissingDataResolution(
                modality, missingness_class, "NONE", "DEGRADED",
                None, None, None, None, None, False,
                tuple(sorted(set(accumulated + ["STRUCTURAL_MASK_AWARE_PATH"]))), None,
            )
        return MissingDataResolution(
            modality, missingness_class, "NONE", "ABSTAIN",
            None, None, None, None, None, False,
            tuple(sorted(set(accumulated + ["STRUCTURAL_UNAVAILABLE_UNSUPPORTED"]))), None,
        )

    if missingness_class == "TRANSIENT_MISSING" and reconstruction_provenance is not None:
        if (
            reconstruction_gap_hours is None
            or not isinstance(reconstruction_gap_hours, (int, float))
            or isinstance(reconstruction_gap_hours, bool)
            or not math.isfinite(reconstruction_gap_hours)
            or reconstruction_gap_hours < 0
        ):
            accumulated.append("RECONSTRUCTION_GAP_INVALID")
        elif reconstruction_gap_hours > policy.maximum_reconstruction_gap_hours[modality]:
            accumulated.append("RECONSTRUCTION_HORIZON_EXCEEDED")
        if reconstruction_evidence_sha256 != policy.reconstruction_evidence_sha256[modality]:
            accumulated.append("RECONSTRUCTION_EVIDENCE_MISMATCH")
        accumulated.extend(
            audit_forecast_time_reconstruction(
                reconstruction_provenance,
                issued_at=issued_at,
                allowed_method_ids={modality: policy.allowed_reconstruction_method_ids[modality]},
                maximum_uncertainty=policy.maximum_reconstruction_uncertainty[modality],
                require_declared_physics=True,
            )
        )
        if not accumulated:
            return MissingDataResolution(
                modality, missingness_class, "RECONSTRUCTED", "DEGRADED",
                None, None, reconstruction_provenance.method_id,
                float(reconstruction_gap_hours),
                float(reconstruction_provenance.normalized_uncertainty),
                False, ("PRIMARY_INPUT_RECOVERED",), reconstruction_evidence_sha256,
            )

    if missingness_class == "TRANSIENT_MISSING" and modality in policy.mask_aware_transient_supported_modalities:
        return MissingDataResolution(
            modality, missingness_class, "NONE", "DEGRADED",
            None, None, None, reconstruction_gap_hours, None, False,
            tuple(sorted(set(accumulated + ["MASK_AWARE_TRANSIENT_PATH"]))), None,
        )

    if missingness_class == "OBSERVED" and primary_observation is None:
        accumulated.append("DECLARED_OBSERVED_WITHOUT_OBSERVATION")
    elif missingness_class == "TRANSIENT_MISSING" and reconstruction_provenance is None:
        accumulated.append("NO_ADMISSIBLE_RECONSTRUCTION")

    return MissingDataResolution(
        modality, missingness_class, "NONE", "ABSTAIN",
        None, None, None, reconstruction_gap_hours, None, False,
        tuple(sorted(set(accumulated or ["NO_ADMISSIBLE_INPUT_PATH"]))), None,
    )
