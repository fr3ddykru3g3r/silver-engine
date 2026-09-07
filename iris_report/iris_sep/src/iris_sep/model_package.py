"""Reloadable IRIS-SEP availability-conditioned model package.

A package contains 15 XGBoost specialists (5 seeds for each of solar, XRS,
proton), frozen feature order, fit prevalence, availability-state fusion
parameters, calibration intercepts, decision thresholds, dependency versions,
and SHA-256 bindings. Loading never trains or calibrates a model.

Runtime deliberately loads raw XGBoost ``Booster`` objects rather than sklearn
wrappers. This keeps serialization independent of sklearn estimator-mixin API
changes while preserving the exact fitted trees.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import xgboost
from xgboost import Booster, DMatrix

PACKAGE_FORMAT = "IRIS_SEP_AVAILABILITY_MODEL_PACKAGE_V1"
STATE_EXPERTS = {
    "FULL": ("SOLAR", "XRS", "PROTON"),
    "NO_XRS": ("SOLAR", "PROTON"),
    "NO_PROTON": ("SOLAR", "XRS"),
    "NO_XRS_OR_PROTON": ("SOLAR",),
}
FAMILY_KEY = {"SOLAR": "solar", "XRS": "xrs", "PROTON": "proton"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1 - 1e-8)
    return np.log(p / (1.0 - p))


def _centered_evidence(probability, prevalence: float, reliability=None, limit: float = 6.0):
    if not 0 < prevalence < 1:
        raise ValueError("fit prevalence must be in (0,1)")
    climate = math.log(prevalence) - math.log1p(-prevalence)
    evidence = np.clip(_logit(probability) - climate, -float(limit), float(limit))
    if reliability is not None:
        r = np.asarray(reliability, dtype=np.float64)
        if r.shape != evidence.shape or not np.isfinite(r).all() or ((r < 0) | (r > 1)).any():
            raise ValueError("invalid reliability")
        evidence = evidence * r
    return evidence


def _family_reliability(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    values = frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    return np.mean(np.isfinite(values), axis=1).astype(np.float64)


def _booster_probability(booster: Booster, frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    numeric = frame.loc[:, names].apply(pd.to_numeric, errors="coerce")
    matrix = DMatrix(numeric, feature_names=list(names), missing=np.nan)
    probability = np.asarray(booster.predict(matrix), dtype=np.float64)
    if probability.ndim != 1 or len(probability) != len(frame):
        raise RuntimeError("specialist booster emitted unexpected prediction shape")
    return probability


def validate_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("format") != PACKAGE_FORMAT:
        raise ValueError("unsupported model package format")
    if manifest.get("target") != "new_sep_10mev_10pfu_within_24h":
        raise ValueError("unexpected target")
    families = manifest.get("feature_families")
    if not isinstance(families, dict) or set(families) != {"solar", "xrs", "proton"}:
        raise ValueError("package must define solar/xrs/proton feature families")
    for names in families.values():
        if not isinstance(names, list) or not names or len(names) != len(set(names)) or not all(isinstance(v, str) and v for v in names):
            raise ValueError("invalid feature family")
    if set(families["solar"]) & set(families["xrs"]) or set(families["solar"]) & set(families["proton"]) or set(families["xrs"]) & set(families["proton"]):
        raise ValueError("feature families overlap")
    seeds = manifest.get("seeds")
    if seeds != [7, 13, 26, 42, 73]:
        raise ValueError("unexpected specialist seeds")
    model_files = manifest.get("model_files")
    if not isinstance(model_files, dict) or set(model_files) != {"solar", "xrs", "proton"}:
        raise ValueError("invalid model file map")
    for family, entries in model_files.items():
        if not isinstance(entries, list) or len(entries) != 5:
            raise ValueError(f"{family} must contain five specialist files")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) < {"seed", "path", "sha256"}:
                raise ValueError("invalid model entry")
            if not isinstance(entry["path"], str) or Path(entry["path"]).is_absolute() or ".." in Path(entry["path"]).parts:
                raise ValueError("model paths must be safe relative paths")
            digest = entry["sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid model SHA-256")
    states = manifest.get("states")
    if not isinstance(states, dict) or set(states) != set(STATE_EXPERTS):
        raise ValueError("availability states incomplete")
    for state, experts in STATE_EXPERTS.items():
        row = states[state]
        if tuple(row.get("experts", [])) != experts:
            raise ValueError(f"state expert mismatch for {state}")
        if not math.isfinite(float(row.get("calibration_intercept"))):
            raise ValueError("invalid calibration intercept")
        thresholds = row.get("thresholds")
        if not isinstance(thresholds, dict) or set(thresholds) != {"MAX_TSS", "POD80_MIN_FAR"}:
            raise ValueError("state thresholds incomplete")
        for value in thresholds.values():
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError("invalid threshold")
        if state != "NO_XRS_OR_PROTON":
            stack = row.get("stack")
            if not isinstance(stack, dict) or len(stack.get("weights", [])) != len(experts):
                raise ValueError("invalid stack parameters")
            if not math.isfinite(float(stack.get("intercept"))) or any(not math.isfinite(float(v)) or float(v) < 0 for v in stack["weights"]):
                raise ValueError("invalid stack parameters")


@dataclass
class LoadedAvailabilityPackage:
    root: Path
    manifest: dict
    models: dict[str, list[Booster]]

    @classmethod
    def load(cls, root: Path, *, enforce_dependency_version: bool = True) -> "LoadedAvailabilityPackage":
        root = Path(root)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("manifest.json missing")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        if enforce_dependency_version and manifest.get("dependencies", {}).get("xgboost") != xgboost.__version__:
            raise ValueError("xgboost version does not match package")
        models: dict[str, list[Booster]] = {}
        for family, entries in manifest["model_files"].items():
            family_models = []
            for entry in entries:
                path = root / entry["path"]
                if not path.is_file() or sha256_file(path) != entry["sha256"]:
                    raise ValueError(f"model integrity failure: {entry['path']}")
                booster = Booster()
                booster.load_model(str(path))
                family_models.append(booster)
            models[family] = family_models
        return cls(root=root, manifest=manifest, models=models)

    def required_features(self, state: str) -> tuple[str, ...]:
        if state not in STATE_EXPERTS:
            raise ValueError("unknown availability state")
        names = []
        for expert in STATE_EXPERTS[state]:
            names.extend(self.manifest["feature_families"][FAMILY_KEY[expert]])
        return tuple(names)

    def _family_probability(self, family: str, frame: pd.DataFrame) -> np.ndarray:
        names = self.manifest["feature_families"][family]
        missing = [name for name in names if name not in frame.columns]
        if missing:
            raise ValueError(f"missing {family} features: {missing[:5]}")
        seed_probability = [_booster_probability(model, frame, names) for model in self.models[family]]
        return np.median(np.stack(seed_probability, axis=0), axis=0).astype(np.float64)

    def predict(self, frame: pd.DataFrame, *, state: str = "FULL") -> np.ndarray:
        if state not in STATE_EXPERTS:
            raise ValueError("unknown availability state")
        if len(frame) == 0:
            raise ValueError("prediction frame must be non-empty")
        fit_prevalence = float(self.manifest["fit_prevalence"])
        limit = float(self.manifest["evidence_limit"])

        raw_solar = self._family_probability("solar", frame)
        if state == "NO_XRS_OR_PROTON":
            raw_state = raw_solar
        else:
            evidence_parts = []
            for expert in STATE_EXPERTS[state]:
                family = FAMILY_KEY[expert]
                if family == "solar":
                    raw = raw_solar
                    reliability = None
                else:
                    raw = self._family_probability(family, frame)
                    reliability = _family_reliability(frame, self.manifest["feature_families"][family])
                evidence_parts.append(_centered_evidence(raw, fit_prevalence, reliability, limit))
            x = np.column_stack(evidence_parts)
            stack = self.manifest["states"][state]["stack"]
            weights = np.asarray(stack["weights"], dtype=np.float64)
            z = float(stack["intercept"]) + x @ weights
            raw_state = _sigmoid(z)

        calibrated = _sigmoid(_logit(raw_state) + float(self.manifest["states"][state]["calibration_intercept"]))
        if not np.isfinite(calibrated).all() or ((calibrated < 0) | (calibrated > 1)).any():
            raise RuntimeError("package emitted invalid probability")
        return calibrated

    def decision(self, frame: pd.DataFrame, *, state: str = "FULL", policy: str = "MAX_TSS") -> dict[str, np.ndarray | float | str]:
        if policy not in ("MAX_TSS", "POD80_MIN_FAR"):
            raise ValueError("unknown threshold policy")
        probability = self.predict(frame, state=state)
        threshold = float(self.manifest["states"][state]["thresholds"][policy])
        return {
            "state": state,
            "policy": policy,
            "threshold": threshold,
            "probability": probability,
            "alert": probability >= threshold,
        }
