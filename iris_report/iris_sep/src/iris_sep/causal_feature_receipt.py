"""Chain-of-custody receipt for causally aggregated IRIS-SEP feature rows.

The receipt binds an exact frozen-schema feature row to:
- the trusted-source authentication result for its raw acquisitions;
- every constituent acquisition receipt digest;
- the exact causal aggregation implementation digest;
- the ordered runtime feature-schema digest.

This closes the gap between 'we downloaded official source bytes' and 'this exact
model input row was derived from those bytes'.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .feature_schema_binding import feature_vector_schema_sha256
from .source_authentication import FORMAT as SOURCE_AUTH_FORMAT


FORMAT = "IRIS_SEP_CAUSAL_FEATURE_DERIVATION_RECEIPT_V1"
AGGREGATOR_PATH = Path(__file__).resolve().parent / "causal_feature_aggregation.py"


class CausalFeatureReceiptError(ValueError):
    """Raised when a feature derivation chain is incomplete or inconsistent."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CausalFeatureReceiptError("feature derivation receipt is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CausalFeatureReceiptError(f"{name} must be timezone-aware")
    return value


def canonical_feature_row_bytes(frame: pd.DataFrame, ordered_features: Sequence[str]) -> bytes:
    """Canonicalize exactly one numeric feature row without losing NaN identity."""
    if not isinstance(frame, pd.DataFrame) or len(frame) != 1:
        raise CausalFeatureReceiptError("exactly one feature row is required")
    names = [str(value) for value in ordered_features]
    if list(frame.columns) != names:
        raise CausalFeatureReceiptError("feature row columns do not exactly match ordered schema")
    values = []
    for name in names:
        raw = pd.to_numeric(pd.Series([frame.iloc[0][name]]), errors="coerce").iloc[0]
        if pd.isna(raw):
            encoded = {"kind": "NaN"}
        else:
            value = float(raw)
            if not math.isfinite(value):
                raise CausalFeatureReceiptError("infinite feature value is not permitted")
            # hex() is an exact, locale-independent float representation.
            encoded = {"kind": "finite", "float_hex": value.hex()}
        values.append({"feature_name": name, "value": encoded})
    return _canonical_json({"features": values})


def build_causal_feature_derivation_receipt(
    *,
    issue_time: datetime,
    feature_row: pd.DataFrame,
    feature_families: Mapping[str, Sequence[str]],
    family_order: Sequence[str],
    source_authentication: Mapping[str, Any],
    acquisition_receipts: Sequence[Mapping[str, Any]],
    aggregator_path: Path = AGGREGATOR_PATH,
) -> dict[str, Any]:
    issue = _aware(issue_time, "issue_time")
    if source_authentication.get("format") != SOURCE_AUTH_FORMAT:
        raise CausalFeatureReceiptError("source authentication receipt format mismatch")
    if source_authentication.get("authenticated_for_prospective_use") is not True:
        raise CausalFeatureReceiptError("source acquisitions are not authenticated for prospective use")
    if source_authentication.get("issue_time") != issue.isoformat():
        raise CausalFeatureReceiptError("source-authentication issue time mismatch")
    auth_sha = source_authentication.get("authentication_receipt_sha256")
    if not _valid_sha(auth_sha):
        raise CausalFeatureReceiptError("source-authentication digest missing")

    ordered_features = [
        str(feature)
        for family in family_order
        for feature in feature_families[str(family)]
    ]
    schema_sha = feature_vector_schema_sha256(feature_families, family_order=family_order)
    row_bytes = canonical_feature_row_bytes(feature_row, ordered_features)
    row_sha = sha256_bytes(row_bytes)

    acquisition_digests = []
    artifact_digests = []
    for receipt in acquisition_receipts:
        if not isinstance(receipt, Mapping):
            raise CausalFeatureReceiptError("acquisition receipt must be a mapping")
        receipt_sha = receipt.get("receipt_sha256")
        artifact_sha = receipt.get("artifact_sha256")
        if not _valid_sha(receipt_sha) or not _valid_sha(artifact_sha):
            raise CausalFeatureReceiptError("acquisition receipt/artifact digest missing")
        acquisition_digests.append(str(receipt_sha))
        artifact_digests.append(str(artifact_sha))
    if not acquisition_digests:
        raise CausalFeatureReceiptError("at least one acquisition receipt is required")
    if len(acquisition_digests) != len(set(acquisition_digests)):
        raise CausalFeatureReceiptError("duplicate acquisition receipt digest")

    payload = {
        "format": FORMAT,
        "issue_time": issue.isoformat(),
        "feature_vector_schema_sha256": schema_sha,
        "feature_row_sha256": row_sha,
        "feature_count": len(ordered_features),
        "source_authentication_sha256": auth_sha,
        "acquisition_receipt_sha256": sorted(acquisition_digests),
        "source_artifact_sha256": sorted(artifact_digests),
        "aggregator_sha256": sha256_file(aggregator_path),
        "aggregator_name": "causal_feature_aggregation.py",
        "retrospective_imputation_used": False,
        "future_observations_used": False,
        "runtime_reconstruction_used": False,
    }
    payload["derivation_receipt_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload


def validate_causal_feature_derivation_receipt(
    *,
    receipt: Mapping[str, Any],
    issue_time: datetime,
    feature_row: pd.DataFrame,
    feature_families: Mapping[str, Sequence[str]],
    family_order: Sequence[str],
    source_authentication: Mapping[str, Any],
    acquisition_receipts: Sequence[Mapping[str, Any]],
    aggregator_path: Path = AGGREGATOR_PATH,
) -> dict[str, Any]:
    rebuilt = build_causal_feature_derivation_receipt(
        issue_time=issue_time,
        feature_row=feature_row,
        feature_families=feature_families,
        family_order=family_order,
        source_authentication=source_authentication,
        acquisition_receipts=acquisition_receipts,
        aggregator_path=aggregator_path,
    )
    if dict(receipt) != rebuilt:
        raise CausalFeatureReceiptError("stored causal feature derivation receipt does not match recomputation")
    return rebuilt
