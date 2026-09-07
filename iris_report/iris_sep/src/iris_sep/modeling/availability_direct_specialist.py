"""Direct raw-feature specialists for IRIS-SEP sensor-availability states.

Fallback V1 combines scalar probabilities from family-specific experts.  That is
simple and auditable, but it can discard interactions between the feature
families that remain available.  V2 instead trains a small classifier directly
on the raw *available* causal predictors for each state:

    NO_XRS            -> SOLAR + PROTON
    NO_PROTON         -> SOLAR + XRS
    NO_XRS_OR_PROTON  -> SOLAR

No unavailable feature is imputed, reconstructed, or silently substituted.
Models are fitted on an explicit fit mask and produce probabilities for the
already-prepared chronological cohort.  Calibration and thresholds remain the
responsibility of the experiment runner so role boundaries stay visible.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .availability_fallback import validate_state


DEFAULT_SEEDS = (7, 13, 26, 42, 73)
STATE_FAMILIES = {
    "FULL": ("SOLAR", "XRS", "PROTON"),
    "NO_XRS": ("SOLAR", "PROTON"),
    "NO_PROTON": ("SOLAR", "XRS"),
    "NO_XRS_OR_PROTON": ("SOLAR",),
}


@dataclass(frozen=True)
class DirectSpecialistConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    n_estimators: int = 500
    learning_rate: float = 0.03
    max_depth: int = 3
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    n_jobs: int = 2

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if any(not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must be integers")
        if self.n_estimators <= 0 or self.max_depth <= 0 or self.n_jobs <= 0:
            raise ValueError("tree counts, depth and n_jobs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.subsample <= 1 or not 0 < self.colsample_bytree <= 1:
            raise ValueError("sampling fractions must be in (0,1]")
        if self.min_child_weight < 0 or self.reg_lambda < 0 or self.reg_alpha < 0:
            raise ValueError("regularization parameters must be nonnegative")


def state_feature_names(
    state: str,
    *,
    solar: Sequence[str],
    xrs: Sequence[str],
    proton: Sequence[str],
) -> tuple[str, ...]:
    """Return the exact ordered raw-feature schema available in ``state``."""
    state = validate_state(state)
    families = {
        "SOLAR": tuple(str(v) for v in solar),
        "XRS": tuple(str(v) for v in xrs),
        "PROTON": tuple(str(v) for v in proton),
    }
    if any(not values for values in families.values()):
        raise ValueError("solar, XRS and proton feature families must all be non-empty")
    ordered: list[str] = []
    for family in STATE_FAMILIES[state]:
        ordered.extend(families[family])
    if len(set(ordered)) != len(ordered):
        raise ValueError("feature families overlap")
    if any(name.lower().startswith("future_") for name in ordered):
        raise ValueError("future columns are forbidden in direct specialists")
    return tuple(ordered)


def feature_schema_sha256(feature_names: Sequence[str]) -> str:
    names = tuple(str(v) for v in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature schema must be non-empty and unique")
    payload = json.dumps(list(names), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_training_inputs(
    frame: pd.DataFrame,
    labels,
    fit_mask,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = tuple(feature_names)
    if not names or any(name not in frame.columns for name in names):
        raise ValueError("all direct-specialist features must exist in frame")
    if any(name.lower().startswith("future_") for name in names):
        raise ValueError("future columns are forbidden")
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    fit = np.asarray(fit_mask, dtype=bool).reshape(-1)
    if len(frame) != len(y) or len(y) != len(fit) or len(y) == 0:
        raise ValueError("frame, labels and fit mask must align")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")
    if fit.sum() < 2 or len(np.unique(y[fit])) != 2:
        raise ValueError("fit rows must contain both classes")
    x = frame.loc[:, names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    # XGBoost handles NaN natively; infinities are never admitted.
    if np.isinf(x).any():
        raise ValueError("infinite raw feature values are forbidden")
    return x, y, fit


def fit_predict_direct_specialist(
    *,
    frame: pd.DataFrame,
    labels,
    fit_mask,
    feature_names: Sequence[str],
    config: DirectSpecialistConfig | None = None,
) -> dict[str, object]:
    """Fit fixed-seed XGBoost specialists and return median probability.

    The exact recipe matches the public NEW-crossing XGBoost benchmark by
    default.  No calibration, threshold fitting, imputation, or outage-time
    retraining occurs here.
    """
    cfg = config or DirectSpecialistConfig()
    names = tuple(feature_names)
    x, y, fit = _validate_training_inputs(frame, labels, fit_mask, names)
    prevalence = float(np.mean(y[fit]))
    if not 0.0 < prevalence < 1.0:
        raise ValueError("fit prevalence must be in (0,1)")
    seed_probability: dict[int, np.ndarray] = {}
    for seed in cfg.seeds:
        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_lambda=cfg.reg_lambda,
            reg_alpha=cfg.reg_alpha,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=(1.0 - prevalence) / prevalence,
            n_jobs=cfg.n_jobs,
            random_state=int(seed),
        )
        model.fit(x[fit], y[fit], verbose=False)
        p = np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
        if p.shape != (len(y),) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise RuntimeError("direct specialist emitted invalid probabilities")
        seed_probability[int(seed)] = p
    stack = np.stack([seed_probability[seed] for seed in cfg.seeds], axis=0)
    probability = np.median(stack, axis=0)
    return {
        "probability": probability,
        "seed_probability": seed_probability,
        "feature_names": names,
        "feature_schema_sha256": feature_schema_sha256(names),
        "seeds": tuple(cfg.seeds),
        "fit_rows": int(fit.sum()),
        "fit_positives": int(y[fit].sum()),
        "imputation_used": False,
        "reconstruction_used": False,
        "runtime_retraining": False,
    }
