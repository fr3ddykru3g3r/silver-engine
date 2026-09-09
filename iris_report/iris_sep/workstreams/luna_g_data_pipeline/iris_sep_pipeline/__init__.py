"""Fail-closed, dependency-light synthetic cohort pipeline for IRIS-SEP.

This package deliberately contains no production data adapters.  It exercises
the integrity boundaries that must hold before any permitted dataset is used.
"""

from .schemas import (
    FeatureRecord,
    IssueRecord,
    Observation,
    TargetRecord,
    CohortUnit,
    canonical_issue_id,
    make_issue,
    validate_issues,
    validate_features,
)
from .cohort import build_targets, build_cohort_units, assign_chronological_roles
from .transforms import FeatureRow, TrainOnlyStandardizer, TransformReceipt, TransformedRow
from .manifests import (
    canonical_json,
    sha256_canonical,
    freeze_manifest,
    write_immutable_manifest,
    verify_manifest,
)

__all__ = [
    "FeatureRow", "FeatureRecord", "IssueRecord", "Observation", "TargetRecord",
    "CohortUnit", "canonical_issue_id", "make_issue", "validate_issues",
    "validate_features", "build_targets", "build_cohort_units",
    "assign_chronological_roles", "TrainOnlyStandardizer", "TransformReceipt",
    "TransformedRow", "canonical_json", "sha256_canonical", "freeze_manifest",
    "write_immutable_manifest", "verify_manifest",
]
