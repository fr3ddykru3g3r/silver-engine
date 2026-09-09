"""Canonical ordered feature-vector schemas for IRIS-SEP runtime binding.

A family-to-feature mapping is not sufficient to describe a model input vector:
model inference is position-sensitive.  This module creates a canonical schema
that binds family order, within-family feature order and each final vector index.
The resulting SHA-256 is suitable for runtime policy, model-package and
prospective-provenance trust anchors.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


FORMAT = "IRIS_SEP_ORDERED_FEATURE_VECTOR_SCHEMA_V1"


class FeatureSchemaBindingError(ValueError):
    """Raised when an ordered feature-vector schema is ambiguous or malformed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FeatureSchemaBindingError("feature schema is not canonical JSON") from exc


def _normalize_family_order(
    feature_families: Mapping[str, Sequence[str]],
    family_order: Sequence[str],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    if not isinstance(feature_families, Mapping) or not feature_families:
        raise FeatureSchemaBindingError("feature_families must be a non-empty mapping")
    if isinstance(family_order, (str, bytes)) or not isinstance(family_order, Sequence):
        raise FeatureSchemaBindingError("family_order must be a sequence")
    order = tuple(str(value) for value in family_order)
    if not order or any(not value for value in order) or len(set(order)) != len(order):
        raise FeatureSchemaBindingError("family_order must contain unique non-empty names")

    normalized: dict[str, tuple[str, ...]] = {}
    seen_features: set[str] = set()
    for raw_family, raw_features in feature_families.items():
        family = str(raw_family)
        if not family:
            raise FeatureSchemaBindingError("feature family cannot be empty")
        if isinstance(raw_features, (str, bytes)) or not isinstance(raw_features, Sequence):
            raise FeatureSchemaBindingError(f"feature list for {family} must be a sequence")
        features = tuple(str(value) for value in raw_features)
        if not features or any(not value for value in features):
            raise FeatureSchemaBindingError(f"feature list for {family} cannot be empty")
        if len(set(features)) != len(features):
            raise FeatureSchemaBindingError(f"duplicate feature within family {family}")
        overlap = seen_features.intersection(features)
        if overlap:
            raise FeatureSchemaBindingError(
                f"feature names must be globally unique; duplicates={sorted(overlap)}"
            )
        seen_features.update(features)
        normalized[family] = features

    if set(order) != set(normalized):
        missing = sorted(set(normalized) - set(order))
        unexpected = sorted(set(order) - set(normalized))
        raise FeatureSchemaBindingError(
            f"family_order must exactly cover feature_families; missing={missing}, unexpected={unexpected}"
        )
    return normalized, order


def build_feature_vector_schema(
    feature_families: Mapping[str, Sequence[str]],
    *,
    family_order: Sequence[str],
) -> dict[str, Any]:
    """Return a canonical position-sensitive feature-vector schema."""
    families, order = _normalize_family_order(feature_families, family_order)
    features: list[dict[str, Any]] = []
    index = 0
    for family in order:
        for feature_name in families[family]:
            features.append(
                {
                    "index": index,
                    "family": family,
                    "feature_name": feature_name,
                }
            )
            index += 1
    return {
        "format": FORMAT,
        "family_order": list(order),
        "feature_count": len(features),
        "features": features,
    }


def feature_vector_schema_sha256(
    feature_families: Mapping[str, Sequence[str]],
    *,
    family_order: Sequence[str],
) -> str:
    """Hash the exact ordered vector schema."""
    schema = build_feature_vector_schema(feature_families, family_order=family_order)
    return hashlib.sha256(_canonical_json(schema)).hexdigest()


def validate_feature_vector_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and verify a serialized schema, rejecting reordered positions."""
    if not isinstance(schema, Mapping) or schema.get("format") != FORMAT:
        raise FeatureSchemaBindingError("unsupported feature-vector schema")
    order = schema.get("family_order")
    rows = schema.get("features")
    count = schema.get("feature_count")
    if not isinstance(order, list) or not isinstance(rows, list) or not isinstance(count, int):
        raise FeatureSchemaBindingError("feature-vector schema fields are malformed")
    if count != len(rows) or count <= 0:
        raise FeatureSchemaBindingError("feature_count does not match feature rows")

    families: dict[str, list[str]] = {str(family): [] for family in order}
    if len(families) != len(order) or any(not family for family in families):
        raise FeatureSchemaBindingError("family_order is invalid")
    for expected_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FeatureSchemaBindingError("feature row must be a mapping")
        if row.get("index") != expected_index:
            raise FeatureSchemaBindingError("feature-vector indices must be contiguous and ordered")
        family = str(row.get("family", ""))
        feature = str(row.get("feature_name", ""))
        if family not in families or not feature:
            raise FeatureSchemaBindingError("feature row has invalid family or feature name")
        families[family].append(feature)

    rebuilt = build_feature_vector_schema(families, family_order=order)
    if dict(schema) != rebuilt:
        raise FeatureSchemaBindingError("serialized feature-vector schema is not canonical")
    return rebuilt
