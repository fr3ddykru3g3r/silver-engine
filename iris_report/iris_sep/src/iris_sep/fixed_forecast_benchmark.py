"""Fixed reference forecaster for the train-only IRIS-SEP missing-data benchmark.

This is not the final IRIS model. It is a deliberately fixed logistic
forecaster used to ask one narrow question: after a forecast-time observation
is hidden, how much does each recovery strategy preserve the *same frozen
forecaster's* output?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from iris_report.iris_sep.workstreams.luna_i_eval_ops.evaluation import (
    apply_calibration,
    fit_intercept_calibration,
    select_tss_threshold,
)
from .missingness_experiment import evaluate_forecast_preservation


_ALLOWED_ROLES = frozenset({"fit", "calibration", "threshold", "score"})


@dataclass(frozen=True)
class FixedReferenceForecaster:
    model: object
    fit_medians: np.ndarray
    supported_features: np.ndarray
    calibration: object
    threshold: float
    seed: int


def _validate(
    values,
    observed_mask,
    structural_unavailable_mask,
    labels,
    roles,
):
    features = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    structural = np.asarray(structural_unavailable_mask, dtype=bool)
    targets = np.asarray(labels)
    role_values = np.asarray(roles, dtype=str)
    if (
        features.ndim != 2
        or features.shape != observed.shape
        or features.shape != structural.shape
        or features.shape[0] == 0
    ):
        raise ValueError(
            "values/masks must be matching non-empty 2-D arrays"
        )
    if targets.shape != (features.shape[0],) or role_values.shape != (
        features.shape[0],
    ):
        raise ValueError("labels/roles must have one value per row")
    if np.any(observed & structural):
        raise ValueError("structurally unavailable cells cannot be observed")
    if not np.isfinite(features[observed]).all():
        raise ValueError("observed values must be finite")
    try:
        targets = targets.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("labels must be binary") from exc
    if not np.isfinite(targets).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("labels must be binary")
    if set(role_values.tolist()) != _ALLOWED_ROLES:
        raise ValueError(
            "roles must be exactly fit/calibration/threshold/score; "
            "locked or extra roles forbidden"
        )
    for role in _ALLOWED_ROLES:
        if len(np.unique(targets[role_values == role])) != 2:
            raise ValueError(f"{role} requires both classes")
    return (
        features,
        observed,
        structural,
        targets.astype(np.int8),
        role_values,
    )


def _fit_medians(features, observed, fit_rows):
    supported = observed[fit_rows].any(axis=0)
    if not supported.any():
        raise ValueError("no features have observed fit support")
    medians = np.zeros(features.shape[1], dtype=np.float64)
    for feature in np.flatnonzero(supported):
        medians[feature] = np.median(
            features[fit_rows & observed[:, feature], feature]
        )
    return medians, supported


def build_design_matrix(
    values,
    observed_mask,
    reconstructed_mask,
    *,
    fit_medians,
    supported_features,
):
    """Build numeric values + observed/reconstructed provenance indicators."""
    features = np.asarray(values, dtype=np.float64)
    observed = np.asarray(observed_mask, dtype=bool)
    reconstructed = np.asarray(reconstructed_mask, dtype=bool)
    medians = np.asarray(fit_medians, dtype=np.float64)
    support = np.asarray(supported_features, dtype=bool)
    if (
        features.ndim != 2
        or features.shape != observed.shape
        or features.shape != reconstructed.shape
        or medians.shape != (features.shape[1],)
        or support.shape != (features.shape[1],)
    ):
        raise ValueError("design inputs have incompatible shapes")
    if np.any(observed & reconstructed):
        raise ValueError("a cell cannot be both observed and reconstructed")
    if (
        not support.any()
        or not np.isfinite(features[observed]).all()
        or not np.isfinite(medians[support]).all()
    ):
        raise ValueError("design values/support are invalid")

    supported_values = features[:, support].copy()
    supported_observed = observed[:, support]
    supported_reconstructed = reconstructed[:, support]
    supported_medians = medians[support]
    usable = supported_observed | supported_reconstructed
    if not np.isfinite(supported_values[usable]).all():
        raise ValueError("observed/reconstructed design values must be finite")
    missing = ~usable
    for feature in range(supported_values.shape[1]):
        supported_values[missing[:, feature], feature] = supported_medians[feature]
    return np.concatenate(
        [
            supported_values,
            supported_observed.astype(np.float64),
            supported_reconstructed.astype(np.float64),
        ],
        axis=1,
    )


def fit_fixed_reference_forecaster(
    *,
    values,
    observed_mask,
    structural_unavailable_mask,
    labels,
    roles,
    seed=20260905,
) -> FixedReferenceForecaster:
    """Fit once on the reference train-only data and freeze calibration/threshold."""
    features, observed, _structural, targets, role_values = _validate(
        values,
        observed_mask,
        structural_unavailable_mask,
        labels,
        roles,
    )
    fit_rows = role_values == "fit"
    calibration_rows = role_values == "calibration"
    threshold_rows = role_values == "threshold"
    medians, support = _fit_medians(features, observed, fit_rows)
    no_reconstruction = np.zeros_like(observed)
    design = build_design_matrix(
        features,
        observed,
        no_reconstruction,
        fit_medians=medians,
        supported_features=support,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    model.fit(design[fit_rows], targets[fit_rows])
    if int(np.max(model[-1].n_iter_)) >= 5000:
        raise RuntimeError("fixed logistic benchmark did not converge")
    raw_logits = model.decision_function(design)
    calibration = fit_intercept_calibration(
        raw_logits[calibration_rows],
        targets[calibration_rows],
        role="validation_calibration",
    )
    probabilities = apply_calibration(raw_logits, calibration)
    threshold_receipt = select_tss_threshold(
        targets[threshold_rows],
        probabilities[threshold_rows],
        role="validation_threshold",
    )
    return FixedReferenceForecaster(
        model=model,
        fit_medians=medians,
        supported_features=support,
        calibration=calibration,
        threshold=float(threshold_receipt.threshold),
        seed=int(seed),
    )


def predict_with_frozen_reference(
    forecaster: FixedReferenceForecaster,
    *,
    values,
    observed_mask,
    reconstructed_mask,
) -> np.ndarray:
    design = build_design_matrix(
        values,
        observed_mask,
        reconstructed_mask,
        fit_medians=forecaster.fit_medians,
        supported_features=forecaster.supported_features,
    )
    return apply_calibration(
        forecaster.model.decision_function(design),
        forecaster.calibration,
    )


def score_recovery_arm(
    forecaster: FixedReferenceForecaster,
    *,
    labels,
    roles,
    reference_probabilities,
    candidate_probabilities,
) -> dict:
    targets = np.asarray(labels)
    role_values = np.asarray(roles, dtype=str)
    if targets.shape != role_values.shape:
        raise ValueError("labels/roles must align")
    score_rows = role_values == "score"
    if not score_rows.any():
        raise ValueError("score role is required")
    return evaluate_forecast_preservation(
        labels=targets[score_rows],
        reference_probabilities=np.asarray(reference_probabilities)[score_rows],
        candidate_probabilities=np.asarray(candidate_probabilities)[score_rows],
        reference_threshold=forecaster.threshold,
        candidate_threshold=forecaster.threshold,
        role="train_only_inner_score",
    )
