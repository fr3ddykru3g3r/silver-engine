"""Load-only package for the promoted IRIS-SEP cross-fitted evidence stack.

This module deliberately contains no training path. Training tools may export a
package, but runtime consumers can only verify, load and score the frozen 15
specialist XGBoost models plus the frozen evidence-stack/calibration/threshold
parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from xgboost import Booster, DMatrix, XGBClassifier


FORMAT = "IRIS_SEP_PROMOTED_MODEL_PACKAGE_V1"
ARCHITECTURE = "IRIS_CROSSFIT_EVIDENCE_STACK_V1"
TARGET = "NEW_GT10MEV_GE10PFU_CROSSING_WITHIN_24H"
FAMILIES = ("SOLAR", "XRS", "PROTON")
MODELS_PER_FAMILY = 5
SERIALIZATION = "XGBOOST_BOOSTER_JSON"


class PromotedModelPackageError(ValueError):
    """Raised when a frozen model package is malformed or fails integrity checks."""


def _canonical_json(value) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PromotedModelPackageError("package metadata is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def feature_schema_sha256(feature_families: Mapping[str, Sequence[str]]) -> str:
    payload = {family: list(feature_families[family]) for family in FAMILIES}
    return sha256_bytes(_canonical_json(payload))


def _validate_feature_families(feature_families: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    if set(feature_families) != set(FAMILIES):
        raise PromotedModelPackageError("feature families must be exactly SOLAR/XRS/PROTON")
    out: dict[str, list[str]] = {}
    all_names: list[str] = []
    for family in FAMILIES:
        names = list(feature_families[family])
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise PromotedModelPackageError(f"invalid feature names for {family}")
        if len(names) != len(set(names)):
            raise PromotedModelPackageError(f"duplicate feature names within {family}")
        out[family] = names
        all_names.extend(names)
    if len(all_names) != len(set(all_names)):
        raise PromotedModelPackageError("feature families overlap")
    return out


def _finite_float(value, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise PromotedModelPackageError(f"{name} must be finite")
    return float(value)


def export_promoted_package(
    *,
    output_dir: Path,
    family_models: Mapping[str, Sequence[XGBClassifier]],
    feature_families: Mapping[str, Sequence[str]],
    fit_prevalence: float,
    stack_intercept: float,
    stack_weights: Sequence[float],
    evidence_limit: float,
    calibration_intercept: float,
    thresholds: Mapping[str, float],
    source_bindings: Mapping[str, str],
    dependency_versions: Mapping[str, str],
    training_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Export one immutable directory and return its manifest.

    The caller is responsible for fitting the models before this function is
    called. This exporter never calls ``fit``. Fitted sklearn wrappers are
    serialized through their native Booster objects to avoid sklearn mixin
    compatibility affecting the on-disk model representation.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise PromotedModelPackageError("output directory must be new and immutable")
    output_dir.mkdir(parents=True)
    feature_families = _validate_feature_families(feature_families)

    prevalence = _finite_float(fit_prevalence, "fit_prevalence")
    if not 0.0 < prevalence < 1.0:
        raise PromotedModelPackageError("fit_prevalence must be in (0,1)")
    weights = [_finite_float(v, "stack weight") for v in stack_weights]
    if len(weights) != 3 or any(v < 0 for v in weights):
        raise PromotedModelPackageError("three nonnegative stack weights required")
    stack_intercept = _finite_float(stack_intercept, "stack_intercept")
    evidence_limit = _finite_float(evidence_limit, "evidence_limit")
    if evidence_limit <= 0:
        raise PromotedModelPackageError("evidence_limit must be positive")
    calibration_intercept = _finite_float(calibration_intercept, "calibration_intercept")
    if set(thresholds) != {"MAX_TSS", "POD80_MIN_FAR"}:
        raise PromotedModelPackageError("both frozen threshold policies required")
    threshold_payload = {k: _finite_float(v, f"threshold {k}") for k, v in thresholds.items()}
    if any(v < 0 or v > 1 for v in threshold_payload.values()):
        raise PromotedModelPackageError("thresholds must be probabilities")

    model_records: dict[str, list[dict[str, object]]] = {}
    for family in FAMILIES:
        models = list(family_models.get(family, ()))
        if len(models) != MODELS_PER_FAMILY:
            raise PromotedModelPackageError(f"{family} must contain exactly {MODELS_PER_FAMILY} specialist models")
        records = []
        for index, model in enumerate(models):
            if not isinstance(model, XGBClassifier):
                raise PromotedModelPackageError("specialists must be XGBClassifier instances")
            relative = Path("models") / family.lower() / f"seed_{index}.json"
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            model.get_booster().save_model(str(destination))
            records.append({
                "index": index,
                "path": relative.as_posix(),
                "sha256": sha256_file(destination),
            })
        model_records[family] = records

    source_payload = {str(k): str(v) for k, v in source_bindings.items()}
    dependency_payload = {str(k): str(v) for k, v in dependency_versions.items()}
    manifest = {
        "format": FORMAT,
        "architecture": ARCHITECTURE,
        "target": TARGET,
        "scope": "DEVELOPMENT_MODEL_PACKAGE_NOT_OPERATIONALLY_CERTIFIED",
        "serialization": SERIALIZATION,
        "models_per_family": MODELS_PER_FAMILY,
        "feature_families": feature_families,
        "feature_schema_sha256": feature_schema_sha256(feature_families),
        "fit_prevalence": prevalence,
        "evidence": {
            "expert_order": list(FAMILIES),
            "reliability": "FINITE_FRACTION_PER_FAMILY_AT_ISSUE_FOR_XRS_AND_PROTON",
            "limit": evidence_limit,
            "stack_intercept": stack_intercept,
            "stack_weights": weights,
        },
        "calibration": {
            "method": "LOGIT_INTERCEPT_ONLY",
            "intercept": calibration_intercept,
        },
        "thresholds": threshold_payload,
        "models": model_records,
        "source_bindings": source_payload,
        "dependency_versions": dependency_payload,
        "training_receipt": dict(training_receipt),
        "runtime_training_allowed": False,
    }
    manifest_bytes = _canonical_json(manifest)
    (output_dir / "manifest.json").write_bytes(manifest_bytes + b"\n")
    receipt = {
        "format": FORMAT,
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "feature_schema_sha256": manifest["feature_schema_sha256"],
        "model_file_count": sum(len(v) for v in model_records.values()),
        "serialization": SERIALIZATION,
        "runtime_training_allowed": False,
    }
    (output_dir / "package_receipt.json").write_bytes(_canonical_json(receipt) + b"\n")
    return manifest


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    expz = np.exp(z[~pos])
    out[~pos] = expz / (1.0 + expz)
    return out


@dataclass
class LoadedPromotedModelPackage:
    root: Path
    manifest: dict[str, object]
    models: dict[str, list[Booster]]

    @property
    def thresholds(self) -> dict[str, float]:
        return {str(k): float(v) for k, v in dict(self.manifest["thresholds"]).items()}

    def predict(self, frame: pd.DataFrame) -> dict[str, np.ndarray]:
        """Score rows without fitting or mutating the package."""
        if not isinstance(frame, pd.DataFrame) or len(frame) == 0:
            raise PromotedModelPackageError("non-empty pandas DataFrame required")
        families = _validate_feature_families(self.manifest["feature_families"])
        missing = [name for family in FAMILIES for name in families[family] if name not in frame.columns]
        if missing:
            raise PromotedModelPackageError(f"required package features missing: {missing[:5]}")

        raw: dict[str, np.ndarray] = {}
        reliability: dict[str, np.ndarray] = {}
        for family in FAMILIES:
            names = families[family]
            values = frame.loc[:, names].apply(pd.to_numeric, errors="coerce")
            dmatrix = DMatrix(values.to_numpy(dtype=np.float64), feature_names=names, missing=np.nan)
            preds = [model.predict(dmatrix) for model in self.models[family]]
            raw[family] = np.median(np.stack(preds, axis=0), axis=0).astype(np.float64)
            reliability[family] = np.mean(np.isfinite(values.to_numpy(dtype=np.float64)), axis=1)

        prevalence = float(self.manifest["fit_prevalence"])
        climate = math.log(prevalence) - math.log1p(-prevalence)
        evidence_cfg = dict(self.manifest["evidence"])
        limit = float(evidence_cfg["limit"])
        evidence = []
        for family in FAMILIES:
            e = np.clip(_logit(np.clip(raw[family], 1e-6, 1 - 1e-6)) - climate, -limit, limit)
            if family in ("XRS", "PROTON"):
                e = e * reliability[family]
            evidence.append(e)
        matrix = np.column_stack(evidence)
        weights = np.asarray(evidence_cfg["stack_weights"], dtype=np.float64)
        z = float(evidence_cfg["stack_intercept"]) + matrix @ weights
        z = z + float(dict(self.manifest["calibration"])["intercept"])
        probability = _sigmoid(z)
        return {
            "probability": probability,
            "raw_solar_probability": raw["SOLAR"],
            "raw_xrs_probability": raw["XRS"],
            "raw_proton_probability": raw["PROTON"],
            "xrs_reliability": reliability["XRS"],
            "proton_reliability": reliability["PROTON"],
        }


def load_promoted_package(root: Path) -> LoadedPromotedModelPackage:
    root = Path(root)
    manifest_path = root / "manifest.json"
    receipt_path = root / "package_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise PromotedModelPackageError("manifest/package receipt missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromotedModelPackageError("invalid package JSON") from exc
    if manifest.get("format") != FORMAT or manifest.get("architecture") != ARCHITECTURE or manifest.get("target") != TARGET:
        raise PromotedModelPackageError("unsupported package identity")
    if manifest.get("serialization") != SERIALIZATION or receipt.get("serialization") != SERIALIZATION:
        raise PromotedModelPackageError("unsupported model serialization")
    if manifest.get("runtime_training_allowed") is not False:
        raise PromotedModelPackageError("runtime-training flag must be false")
    if receipt.get("manifest_sha256") != sha256_file(manifest_path):
        raise PromotedModelPackageError("manifest digest mismatch")
    families = _validate_feature_families(manifest.get("feature_families", {}))
    if manifest.get("feature_schema_sha256") != feature_schema_sha256(families):
        raise PromotedModelPackageError("feature schema digest mismatch")
    if int(manifest.get("models_per_family", -1)) != MODELS_PER_FAMILY:
        raise PromotedModelPackageError("unexpected specialist count")

    models_payload = manifest.get("models")
    if not isinstance(models_payload, Mapping) or set(models_payload) != set(FAMILIES):
        raise PromotedModelPackageError("model records malformed")
    loaded: dict[str, list[Booster]] = {}
    for family in FAMILIES:
        records = list(models_payload[family])
        if len(records) != MODELS_PER_FAMILY:
            raise PromotedModelPackageError(f"wrong model count for {family}")
        family_models: list[Booster] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise PromotedModelPackageError("invalid model record")
            relative = Path(str(record.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise PromotedModelPackageError("unsafe model path")
            model_path = root / relative
            if not model_path.is_file() or sha256_file(model_path) != record.get("sha256"):
                raise PromotedModelPackageError("model file digest mismatch")
            model = Booster()
            model.load_model(str(model_path))
            family_models.append(model)
        loaded[family] = family_models

    evidence = manifest.get("evidence")
    if not isinstance(evidence, Mapping) or list(evidence.get("expert_order", ())) != list(FAMILIES):
        raise PromotedModelPackageError("evidence configuration malformed")
    weights = list(evidence.get("stack_weights", ()))
    if len(weights) != 3 or any(not math.isfinite(float(v)) or float(v) < 0 for v in weights):
        raise PromotedModelPackageError("invalid stack weights")
    _finite_float(evidence.get("stack_intercept"), "stack intercept")
    if _finite_float(evidence.get("limit"), "evidence limit") <= 0:
        raise PromotedModelPackageError("invalid evidence limit")
    prevalence = _finite_float(manifest.get("fit_prevalence"), "fit prevalence")
    if not 0 < prevalence < 1:
        raise PromotedModelPackageError("invalid fit prevalence")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("method") != "LOGIT_INTERCEPT_ONLY":
        raise PromotedModelPackageError("calibration binding malformed")
    _finite_float(calibration.get("intercept"), "calibration intercept")
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {"MAX_TSS", "POD80_MIN_FAR"}:
        raise PromotedModelPackageError("threshold binding malformed")
    for value in thresholds.values():
        v = _finite_float(value, "threshold")
        if not 0 <= v <= 1:
            raise PromotedModelPackageError("threshold outside [0,1]")

    return LoadedPromotedModelPackage(root=root, manifest=manifest, models=loaded)
