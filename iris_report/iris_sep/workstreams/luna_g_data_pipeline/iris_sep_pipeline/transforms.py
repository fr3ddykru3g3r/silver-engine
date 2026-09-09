"""Train-only numeric transformation with a hashed fit receipt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable, Mapping

from .errors import DuplicateRecordError, PipelineError, ProtectedDataError


@dataclass(frozen=True)
class FeatureRow:
    issue_id: str
    role: str
    values: Mapping[str, float | None]
    origin: str = "synthetic"

    def __post_init__(self) -> None:
        if self.origin != "synthetic":
            raise ProtectedDataError("only synthetic feature rows are accepted")
        if not self.issue_id or not self.values:
            raise PipelineError("issue_id and non-empty values are required")
        for name, value in self.values.items():
            if not name or (value is not None and not math.isfinite(float(value))):
                raise PipelineError("feature names must be non-empty and values finite or None")


@dataclass(frozen=True)
class TransformReceipt:
    fit_role: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    missing_fill: tuple[float, ...]
    training_issue_ids_sha256: str
    receipt_sha256: str


@dataclass(frozen=True)
class TransformedRow:
    issue_id: str
    role: str
    values: tuple[float, ...]
    observed_mask: tuple[bool, ...]


def _hash_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class TrainOnlyStandardizer:
    """Median-impute and standardize using training rows only."""

    def __init__(self) -> None:
        self.receipt: TransformReceipt | None = None

    def fit(self, rows: Iterable[FeatureRow]) -> TransformReceipt:
        materialized = list(rows)
        if not materialized or any(row.role != "train" for row in materialized):
            raise ProtectedDataError("transform fitting accepts train-role rows only")
        ids = [row.issue_id for row in materialized]
        if len(ids) != len(set(ids)):
            raise DuplicateRecordError("duplicate issue_id in transform fit")
        names = tuple(sorted(materialized[0].values))
        if any(tuple(sorted(row.values)) != names for row in materialized):
            raise PipelineError("all transform rows must use the same feature manifest")
        columns: list[list[float]] = []
        for name in names:
            observed = [float(row.values[name]) for row in materialized if row.values[name] is not None]
            if not observed:
                raise PipelineError(f"feature {name!r} has no observed training values")
            columns.append(observed)
        fills = tuple(float(sorted(values)[len(values) // 2]) for values in columns)
        dense = [
            [float(row.values[name]) if row.values[name] is not None else fills[index] for index, name in enumerate(names)]
            for row in materialized
        ]
        means = tuple(sum(row[index] for row in dense) / len(dense) for index in range(len(names)))
        scales = []
        for index in range(len(names)):
            variance = sum((row[index] - means[index]) ** 2 for row in dense) / len(dense)
            scales.append(math.sqrt(variance) if variance > 0 else 1.0)
        core = {
            "fit_role": "train",
            "feature_names": names,
            "means": means,
            "scales": tuple(scales),
            "missing_fill": fills,
            "training_issue_ids_sha256": _hash_json(sorted(ids)),
        }
        self.receipt = TransformReceipt(**core, receipt_sha256=_hash_json(core))
        return self.receipt

    def transform(self, rows: Iterable[FeatureRow]) -> list[TransformedRow]:
        if self.receipt is None:
            raise PipelineError("fit must be called before transform")
        result: list[TransformedRow] = []
        names = self.receipt.feature_names
        for row in rows:
            if tuple(sorted(row.values)) != names:
                raise PipelineError("row does not match fitted feature manifest")
            observed = tuple(row.values[name] is not None for name in names)
            dense = [
                float(row.values[name]) if row.values[name] is not None else self.receipt.missing_fill[index]
                for index, name in enumerate(names)
            ]
            values = tuple(
                (dense[index] - self.receipt.means[index]) / self.receipt.scales[index]
                for index in range(len(names))
            )
            result.append(TransformedRow(row.issue_id, row.role, values, observed))
        return result
