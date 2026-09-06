"""Fail-closed tuning-time interfaces for IRIS-SEP baseline adapters.

This file deliberately contains no dataset loader and no model implementation.
It validates only the boundary between a future baseline runner and the frozen
benchmark. In particular, it cannot accept locked-test rows, labels, or
predictions during tuning. The primary agent owns the separate final evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


class BaselineContractError(ValueError):
    """Raised when a baseline run would violate the tuning-time contract."""


TUNING_ROLES = frozenset(
    {
        "train",
        "validation_monitor",
        "validation_calibration",
        "validation_threshold",
    }
)
LOCKED_TEST_ROLE = "locked_test"
REQUIRED_PREDICTION_FIELDS = frozenset(
    {
        "row_id",
        "issue_time_utc",
        "partition",
        "model_name",
        "seed",
        "p_sep_10mev_10pfu",
        "threshold_used",
    }
)
FORBIDDEN_OUTCOME_FIELDS = frozenset(
    {
        "y_true",
        "label",
        "future_sep_label",
        "future_operational_sep_label",
        "future_osep_label",
        "future_gsep_label",
        "locked_test_outcome",
    }
)


@dataclass(frozen=True)
class BaselineRunSpec:
    """Predeclared identity for one tuning-time baseline run."""

    model_name: str
    seed: int
    feature_set: str
    target: str = "new_sep_10mev_10pfu_within_24h"
    phase: str = "tuning"

    def validate(self) -> None:
        if not self.model_name.strip():
            raise BaselineContractError("model_name must be nonempty")
        if not isinstance(self.seed, int):
            raise BaselineContractError("seed must be an integer")
        if not self.feature_set.strip():
            raise BaselineContractError("feature_set must be nonempty")
        if self.phase != "tuning":
            raise BaselineContractError(
                "this interface is tuning-only; use the primary agent's separate final evaluator"
            )
        if self.target != "new_sep_10mev_10pfu_within_24h":
            raise BaselineContractError(
                "baseline interface currently guards the primary target only"
            )


def assert_no_locked_test_access(
    rows: Iterable[Mapping[str, Any]], *, phase: str = "tuning"
) -> None:
    """Reject any locked-test row, regardless of whether labels are requested.

    Rejecting test *features* as well as outcomes during tuning prevents a
    cohort-specific preprocessing or threshold decision from being hidden in a
    supposedly label-free preview. Final evaluation intentionally does not use
    this function.
    """

    if phase != "tuning":
        raise BaselineContractError(
            "assert_no_locked_test_access is only valid for tuning"
        )
    for index, row in enumerate(rows):
        if str(row.get("partition", "")).strip() == LOCKED_TEST_ROLE:
            raise BaselineContractError(
                f"locked_test row encountered during tuning at index {index}"
            )


def assert_tuning_scope(rows: Iterable[Mapping[str, Any]]) -> None:
    """Alias used by runners before fitting or threshold/calibration work."""

    assert_no_locked_test_access(rows, phase="tuning")


def assert_unique_row_ids(rows: Iterable[Mapping[str, Any]]) -> None:
    """Reject missing or duplicate identities before a baseline fit."""

    seen: set[str] = set()
    for index, row in enumerate(rows):
        row_id = str(row.get("row_id", "")).strip()
        if not row_id:
            raise BaselineContractError(f"row {index} has no row_id")
        if row_id in seen:
            raise BaselineContractError(f"duplicate row_id: {row_id}")
        seen.add(row_id)


def validate_prediction_records(
    records: Sequence[Mapping[str, Any]], *, allow_outcome_fields: bool = False
) -> None:
    """Validate a tuning-time prediction frame without loading any labels.

    Outcome columns are rejected by default so a serialized prediction frame
    cannot accidentally become a tuning input. Labels for validation metrics
    belong in a separate, role-scoped evaluation table.
    """

    if not records:
        raise BaselineContractError("prediction frame is empty")
    seen: set[tuple[str, int]] = set()
    for index, record in enumerate(records):
        missing = sorted(REQUIRED_PREDICTION_FIELDS.difference(record))
        if missing:
            raise BaselineContractError(
                f"prediction row {index} is missing fields: {', '.join(missing)}"
            )
        if str(record.get("partition", "")).strip() not in TUNING_ROLES:
            raise BaselineContractError(
                f"prediction row {index} is not a tuning partition"
            )
        if not allow_outcome_fields and FORBIDDEN_OUTCOME_FIELDS.intersection(record):
            fields = sorted(FORBIDDEN_OUTCOME_FIELDS.intersection(record))
            raise BaselineContractError(
                f"prediction frame contains outcome fields: {', '.join(fields)}"
            )
        row_id = str(record["row_id"]).strip()
        if not row_id:
            raise BaselineContractError(f"prediction row {index} has empty row_id")
        try:
            seed = int(record["seed"])
        except (TypeError, ValueError) as exc:
            raise BaselineContractError(f"prediction row {index} has invalid seed") from exc
        key = (row_id, seed)
        if key in seen:
            raise BaselineContractError(
                f"duplicate row_id/seed in prediction frame: {row_id}/{seed}"
            )
        seen.add(key)
        for field in ("p_sep_10mev_10pfu", "threshold_used"):
            value = record[field]
            if value is None:
                raise BaselineContractError(
                    f"prediction row {index} has null {field}"
                )
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise BaselineContractError(
                    f"prediction row {index} has nonnumeric {field}"
                ) from exc
            if not math.isfinite(numeric_value) or not 0.0 <= numeric_value <= 1.0:
                raise BaselineContractError(
                    f"prediction row {index} has out-of-range {field}"
                )


def assert_causal_feature_names(feature_names: Iterable[str]) -> None:
    """Guard obvious future/label columns at the adapter boundary.

    The authoritative feature manifest remains the source of truth. This small
    heuristic is a second tripwire, not a replacement for a timestamp audit.
    """

    forbidden_tokens = (
        "future_",
        "_label",
        "target",
        "outcome",
        "onset_actual",
    )
    offenders = sorted(
        {
            str(name)
            for name in feature_names
            if any(token in str(name).lower() for token in forbidden_tokens)
        }
    )
    if offenders:
        raise BaselineContractError(
            "feature names look label/future-derived: " + ", ".join(offenders)
        )

