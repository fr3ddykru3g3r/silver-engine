"""Isolated, label-free AIA auxiliary-pretraining interface.

The Harvard AARP parameter table is a useful compact AIA-derived feature
resource, but the raw AIA archive is approximately 9.5 TB.  This module does
not download, enumerate, or depend on that archive.  It defines a small
metadata-and-feature contract for an optional self-supervised AIA encoder and
rejects SEP labels, HMI bridge keys, locked-test rows, and future-derived
fields at the boundary.

The output of this interface is an encoder initialization receipt, never an
SEP forecast.  A later SEP fine-tuning job must reset task heads and use the
independent frozen benchmark/identity contracts owned by the primary agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "iris-sep-luna-d-aia-pretraining-v1"
RECEIPT_TYPE = "IRIS_SEP_AIA_AUXILIARY_PRETRAINING"
RAW_AIA_ARCHIVE_APPROX_BYTES = 9_500_000_000_000
ALLOWED_PARTITIONS = frozenset({"aia_pretraining_train", "aia_pretraining_validation"})
ALLOWED_TARGET_KINDS = frozenset(
    {
        "masked_feature_reconstruction",
        "next_aia_observation",
        "instrument_quality_control",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_KEY_TOKENS = (
    "future",
    "_label",
    "label_",
    "outcome",
    "y_true",
    "locked_test",
    "harpnum",
    "t_rec_tai",
    "hmi",
    "fusion",
    "calibration",
    "threshold",
)
_ALLOWED_TARGET_KEYS = frozenset({"pretraining_target", "pretraining_target_kind"})


class AIAContractError(ValueError):
    """Raised when a pretraining record is not label-free and external-only."""


@dataclass(frozen=True)
class AIApretrainingSpec:
    """Predeclared self-supervised interface; no SEP label is part of it."""

    task_id: str = "aia_masked_feature_reconstruction_v1"
    feature_schema_version: str = "aarp-176-variables-v1"
    feature_count: int = 176
    target_kind: str = "masked_feature_reconstruction"
    target_namespace: str = "aia_self_supervised"
    sequence_length_hours: int = 7

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "input_schema": {
                "feature_schema_version": self.feature_schema_version,
                "feature_count": self.feature_count,
                "sequence_length_hours": self.sequence_length_hours,
                "channels": [
                    "aia_feature_values",
                    "aia_observation_mask",
                    "aia_time_since_observation_hours",
                ],
                "causal_sep_target": False,
            },
            "target_kind": self.target_kind,
            "target_namespace": self.target_namespace,
            "sep_labels_used": False,
            "locked_test_accessed": False,
            "archive_policy": {
                "raw_aia_archive_approx_bytes": RAW_AIA_ARCHIVE_APPROX_BYTES,
                "raw_aia_archive_size_is_approximate": True,
                "raw_archive_download": "PROHIBITED",
                "allowed_artifacts": [
                    "published AARP parameter table metadata",
                    "small selected feature rows with verified source hashes",
                    "format/provenance smoke samples",
                ],
                "raw_image_training_claim": "FORBIDDEN_WITHOUT_VERIFIED_MULTI_FRAME_ARCHIVE",
            },
            "transfer_contract": {
                "transferable": ["aia_encoder_weights", "train_only_aia_normalization_stats"],
                "not_transferable": [
                    "sep_prediction_head",
                    "sep_labels",
                    "sep_calibration",
                    "sep_threshold",
                    "published_nci_probability_as_label",
                ],
                "fine_tuning_requires": [
                    "reset_task_heads",
                    "frozen_hmi_benchmark_contract",
                    "exact_identity_bridge_if_hmi_fusion_is_attempted",
                ],
            },
        }


def default_pretraining_spec() -> dict[str, Any]:
    """Return the JSON-shaped default interface without accessing any archive."""

    return AIApretrainingSpec().as_dict()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise AIAContractError(f"{field} must be a nonempty, trimmed string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    value = _require_text(value, field).lower()
    if _SHA256.fullmatch(value) is None:
        raise AIAContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_utc(value: Any, field: str) -> str:
    text = _require_text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AIAContractError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AIAContractError(f"{field} must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    if parsed.microsecond:
        raise AIAContractError(f"{field} has fractional seconds")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AIAContractError("provenance must be an object")
    required = ("source_uri", "source_version", "source_sha256", "source_row_id", "retrieved_at_utc")
    missing = [field for field in required if field not in value]
    if missing:
        raise AIAContractError("provenance missing fields: " + ", ".join(missing))
    out = dict(value)
    _require_text(out["source_uri"], "provenance.source_uri")
    _require_text(out["source_version"], "provenance.source_version")
    _require_sha256(out["source_sha256"], "provenance.source_sha256")
    _require_text(out["source_row_id"], "provenance.source_row_id")
    out["retrieved_at_utc"] = _canonical_utc(out["retrieved_at_utc"], "provenance.retrieved_at_utc")
    return out


def _assert_external_only_metadata(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if key_text in _ALLOWED_TARGET_KEYS:
                # These two keys are allowed only as the declared
                # self-supervised target interface; their values are checked
                # separately and cannot contain SEP labels.
                if isinstance(nested, str) and any(
                    token in nested.lower() for token in ("sep", "future", "label", "outcome")
                ):
                    raise AIAContractError(f"{path}.{key} names a SEP/outcome target")
                continue
            if any(token in key_lower for token in _FORBIDDEN_KEY_TOKENS):
                raise AIAContractError(f"{path}.{key} is forbidden for AIA pretraining")
            _assert_external_only_metadata(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_external_only_metadata(nested, f"{path}[{index}]")


def _validate_matrix(value: Any, field: str) -> tuple[list[list[float]], tuple[int, int]]:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise AIAContractError(f"{field} must be a nonempty rectangular list of rows")
    width = len(value[0])
    normalized: list[list[float]] = []
    for row_index, row in enumerate(value):
        if len(row) != width:
            raise AIAContractError(f"{field} is not rectangular at row {row_index}")
        normalized_row: list[float] = []
        for column_index, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise AIAContractError(f"{field}[{row_index}][{column_index}] is not finite")
            normalized_row.append(float(item))
        normalized.append(normalized_row)
    return normalized, (len(normalized), width)


def _validate_mask(value: Any, shape: tuple[int, int]) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise AIAContractError("observed_mask shape does not match sequence")
    normalized: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != shape[1]:
            raise AIAContractError("observed_mask must have the same rectangular shape as sequence")
        normalized_row: list[int] = []
        for column_index, item in enumerate(row):
            if isinstance(item, bool):
                normalized_row.append(int(item))
            elif isinstance(item, int) and item in {0, 1}:
                normalized_row.append(item)
            else:
                raise AIAContractError(
                    f"observed_mask[{row_index}][{column_index}] must be 0 or 1"
                )
        normalized.append(normalized_row)
    return normalized


def validate_pretraining_spec(spec: Mapping[str, Any] | AIApretrainingSpec) -> dict[str, Any]:
    """Validate the predeclared self-supervised spec and return a copy."""

    raw = spec.as_dict() if isinstance(spec, AIApretrainingSpec) else dict(spec)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise AIAContractError("unsupported AIA pretraining schema version")
    _require_text(raw.get("task_id"), "task_id")
    _require_text(raw.get("target_namespace"), "target_namespace")
    if raw["target_namespace"] != "aia_self_supervised":
        raise AIAContractError("target_namespace must be aia_self_supervised")
    target_kind = _require_text(raw.get("target_kind"), "target_kind")
    if target_kind not in ALLOWED_TARGET_KINDS:
        raise AIAContractError(f"unsupported target_kind: {target_kind}")
    if raw.get("sep_labels_used") is not False:
        raise AIAContractError("SEP labels must be false for AIA pretraining")
    if raw.get("locked_test_accessed") is not False:
        raise AIAContractError("locked_test_accessed must be false")
    input_schema = raw.get("input_schema")
    if not isinstance(input_schema, Mapping):
        raise AIAContractError("input_schema must be an object")
    feature_count = input_schema.get("feature_count")
    if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count <= 0:
        raise AIAContractError("input_schema.feature_count must be a positive integer")
    sequence_length = input_schema.get("sequence_length_hours")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length <= 0:
        raise AIAContractError("input_schema.sequence_length_hours must be a positive integer")
    policy = raw.get("archive_policy")
    if not isinstance(policy, Mapping) or policy.get("raw_archive_download") != "PROHIBITED":
        raise AIAContractError("raw AIA archive download must be PROHIBITED")
    if policy.get("raw_aia_archive_size_is_approximate") is not True:
        raise AIAContractError("raw archive size must remain marked approximate")
    transfer = raw.get("transfer_contract")
    if not isinstance(transfer, Mapping):
        raise AIAContractError("transfer_contract is required")
    for forbidden in ("sep_prediction_head", "sep_labels", "sep_calibration", "sep_threshold"):
        if forbidden not in set(transfer.get("not_transferable", [])):
            raise AIAContractError(f"transfer contract must forbid {forbidden}")
    return raw


def validate_pretraining_records(
    rows: Iterable[Mapping[str, Any]], spec: Mapping[str, Any] | AIApretrainingSpec | None = None
) -> dict[str, Any]:
    """Validate small AIA self-supervised examples without reading raw archives."""

    normalized_spec = validate_pretraining_spec(spec or AIApretrainingSpec())
    input_schema = normalized_spec["input_schema"]
    feature_count = int(input_schema["feature_count"])
    sequence_length = int(input_schema["sequence_length_hours"])
    target_kind = normalized_spec["target_kind"]
    records = list(rows)
    if not records:
        raise AIAContractError("pretraining record set is empty")
    seen: set[str] = set()
    dimensions: set[tuple[int, int]] = set()
    source_hashes: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise AIAContractError(f"record {index} is not an object")
        _assert_external_only_metadata(row, f"record[{index}]")
        required = {
            "example_id",
            "aia_source_id",
            "issue_time_utc",
            "partition",
            "sequence",
            "observed_mask",
            "pretraining_target_kind",
            "pretraining_target",
            "provenance",
        }
        missing = sorted(required - set(row))
        if missing:
            raise AIAContractError(f"record {index} missing fields: {', '.join(missing)}")
        example_id = _require_text(row["example_id"], f"record[{index}].example_id")
        if example_id in seen:
            raise AIAContractError(f"duplicate example_id: {example_id}")
        seen.add(example_id)
        _require_text(row["aia_source_id"], f"record[{index}].aia_source_id")
        _canonical_utc(row["issue_time_utc"], f"record[{index}].issue_time_utc")
        if row["partition"] not in ALLOWED_PARTITIONS:
            raise AIAContractError(
                f"record {index} must use an external AIA pretraining partition"
            )
        if row["pretraining_target_kind"] != target_kind:
            raise AIAContractError(f"record {index} target_kind differs from the predeclared spec")
        sequence, shape = _validate_matrix(row["sequence"], f"record[{index}].sequence")
        if shape != (sequence_length, feature_count):
            raise AIAContractError(
                f"record {index} sequence shape {shape} != {(sequence_length, feature_count)}"
            )
        mask = _validate_mask(row["observed_mask"], shape)
        target = row["pretraining_target"]
        if target_kind == "masked_feature_reconstruction":
            if not isinstance(target, list) or len(target) != shape[0]:
                raise AIAContractError(f"record {index} reconstruction target has wrong shape")
            for row_index, target_row in enumerate(target):
                if not isinstance(target_row, list) or len(target_row) != shape[1]:
                    raise AIAContractError(f"record {index} reconstruction target is not rectangular")
                for column_index, item in enumerate(target_row):
                    if item is None:
                        continue
                    if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                        raise AIAContractError(
                            f"record {index} reconstruction target has nonfinite value at "
                            f"{row_index},{column_index}"
                        )
                    # A target is only meaningful where the source value was
                    # masked.  This also keeps the pretraining objective
                    # explicit rather than allowing a copy-through identity.
                    if mask[row_index][column_index] == 1:
                        raise AIAContractError(
                            f"record {index} target leaks an observed value at {row_index},{column_index}"
                        )
        elif target_kind == "next_aia_observation":
            if not isinstance(target, list) or len(target) != feature_count:
                raise AIAContractError(f"record {index} next-observation target has wrong shape")
            for column_index, item in enumerate(target):
                if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                    raise AIAContractError(f"record {index} next-observation target[{column_index}] is not finite")
        elif target_kind == "instrument_quality_control":
            if isinstance(target, bool):
                target_numeric = int(target)
            elif isinstance(target, int) and target in {0, 1}:
                target_numeric = target
            else:
                raise AIAContractError(f"record {index} quality-control target must be 0 or 1")
            _ = target_numeric
        provenance = _validate_provenance(row["provenance"])
        source_hashes.add(provenance["source_sha256"])
        dimensions.add(shape)

    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "task_id": normalized_spec["task_id"],
        "target_kind": target_kind,
        "example_count": len(records),
        "dimensions": [{"sequence_length": item[0], "feature_count": item[1]} for item in sorted(dimensions)],
        "source_sha256_values": sorted(source_hashes),
        "sep_labels_used": False,
        "locked_test_accessed": False,
        "raw_aia_archive_downloaded": False,
        "fusion_or_hmi_join_performed": False,
    }


def build_pretraining_receipt(
    validation: Mapping[str, Any],
    *,
    generated_at_utc: str,
    spec: Mapping[str, Any] | AIApretrainingSpec | None = None,
) -> dict[str, Any]:
    """Build an immutable receipt for a label-free auxiliary pretraining batch."""

    normalized_spec = validate_pretraining_spec(spec or AIApretrainingSpec())
    generated = _canonical_utc(generated_at_utc, "generated_at_utc")
    body = {
        "receipt_type": RECEIPT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "workstream": "luna_d",
        "generated_at_utc": generated,
        "status": validation.get("status", "FAIL"),
        "task_id": normalized_spec["task_id"],
        "target_namespace": normalized_spec["target_namespace"],
        "validation": dict(validation),
        "archive_policy": normalized_spec["archive_policy"],
        "transfer_contract": normalized_spec["transfer_contract"],
        "scientific_boundaries": {
            "sep_labels_used": False,
            "locked_test_accessed": False,
            "raw_aia_archive_downloaded": False,
            "hmi_fusion": "disabled_until_exact_identity_bridge",
            "published_nci_probabilities": "not_labels_or_training_targets",
        },
    }
    body["receipt_sha256"] = sha256_json(body)
    return body


def build_auxiliary_pretraining_interface() -> dict[str, Any]:
    """Return the interface consumed by a future Colab pretraining runner."""

    spec = default_pretraining_spec()
    return {
        "interface_type": "aia_encoder_initialization_only",
        "spec": spec,
        "runner_requirements": {
            "runtime": "Colab GPU optional; CPU smoke test supported",
            "download_policy": "do not download raw AIA archive",
            "accepted_inputs": [
                "small verified AARP feature rows",
                "AIA observation masks",
                "time-since-observation channels",
            ],
            "rejected_inputs": [
                "SEP labels",
                "Future_OSEP_label or Future_GSEP_label",
                "locked_test rows or outcomes",
                "HMI keys before the identity bridge",
                "published probabilities treated as labels",
            ],
            "required_receipt_fields": [
                "source_sha256_values",
                "sep_labels_used",
                "locked_test_accessed",
                "raw_aia_archive_downloaded",
            ],
        },
    }


__all__ = [
    "AIAContractError",
    "AIApretrainingSpec",
    "ALLOWED_PARTITIONS",
    "ALLOWED_TARGET_KINDS",
    "RAW_AIA_ARCHIVE_APPROX_BYTES",
    "SCHEMA_VERSION",
    "build_auxiliary_pretraining_interface",
    "build_pretraining_receipt",
    "default_pretraining_spec",
    "sha256_json",
    "validate_pretraining_records",
    "validate_pretraining_spec",
]
