"""Small, framework-independent contracts for the Luna primary model.

This module deliberately contains no data loading or project-specific feature
names.  It is the boundary between a forecast-time adapter and a model.  The
NumPy implementation is executable in a clean Python/Colab runtime; a neural
network may consume the prepared values and masks without changing the loss or
uncertainty semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


TASKS = ("occurrence", "peak", "onset")


@dataclass(frozen=True)
class ModelConfig:
    """Task activation policy; occurrence is mandatory and primary by default."""

    tasks: tuple[str, ...] = ("occurrence",)
    weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tasks or "occurrence" not in self.tasks:
            raise ValueError("occurrence must remain active as the primary task")
        unknown = set(self.tasks).difference(TASKS)
        if unknown:
            raise ValueError(f"unknown task(s): {sorted(unknown)}")
        if len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be unique")
        for name, weight in self.weights.items():
            if name not in TASKS or not np.isfinite(weight) or weight <= 0:
                raise ValueError(f"task weight for {name!r} must be finite and positive")

    @property
    def active_tasks(self) -> tuple[str, ...]:
        return self.tasks

    def weight(self, task: str) -> float:
        return float(self.weights.get(task, 1.0))


@dataclass(frozen=True)
class FeatureBatch:
    values: dict[str, np.ndarray]
    feature_masks: dict[str, np.ndarray]
    modality_masks: dict[str, np.ndarray]
    all_missing: np.ndarray


def prepare_feature_batch(
    values: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray]
) -> FeatureBatch:
    """Sanitize feature tensors while retaining a mask for every feature.

    Values with missing or non-finite observations are zeroed only at the
    adapter boundary.  The corresponding feature-level mask is passed along,
    so zero is never interpreted as an observed physical measurement.
    """

    if not values or set(values) != set(masks):
        raise ValueError("values and masks must contain the same non-empty modalities")
    clean_values: dict[str, np.ndarray] = {}
    feature_masks: dict[str, np.ndarray] = {}
    modality_masks: dict[str, np.ndarray] = {}
    batch_size: int | None = None
    for modality in values:
        array = np.asarray(values[modality], dtype=float)
        raw_mask = np.asarray(masks[modality], dtype=bool)
        if array.ndim != 3 or raw_mask.shape != array.shape:
            raise ValueError(f"{modality}: expected values and masks shaped [batch,time,feature]")
        if batch_size is None:
            batch_size = array.shape[0]
        elif array.shape[0] != batch_size:
            raise ValueError("all modalities must have the same batch dimension")
        effective = raw_mask & np.isfinite(array)
        clean_values[modality] = np.where(effective, array, 0.0)
        feature_masks[modality] = effective
        modality_masks[modality] = np.any(effective, axis=(1, 2))
    assert batch_size is not None
    stacked = np.stack(tuple(modality_masks.values()), axis=1)
    return FeatureBatch(clean_values, feature_masks, modality_masks, ~np.any(stacked, axis=1))


@dataclass(frozen=True)
class FusionDiagnostics:
    observed_count: np.ndarray
    observed_fraction: np.ndarray
    all_missing: np.ndarray


def fuse_modalities(
    representations: np.ndarray,
    observed: np.ndarray,
    *,
    fallback: np.ndarray,
) -> tuple[np.ndarray, FusionDiagnostics]:
    """Fuse only observed modality representations with an explicit fallback.

    The denominator is the number of observed modalities, not the configured
    number of modalities.  This prevents missing inputs from biasing the
    representation toward zero.  All-missing rows use the calibrated fallback
    and must be abstained by the downstream decision policy.
    """

    rep = np.asarray(representations, dtype=float)
    seen = np.asarray(observed, dtype=bool)
    fallback = np.asarray(fallback, dtype=float)
    if rep.ndim != 3 or seen.shape != rep.shape[:2] or fallback.shape != (rep.shape[2],):
        raise ValueError("representations must be [batch,modality,dim], with matching masks and fallback [dim]")
    if not np.all(np.isfinite(rep)) or not np.all(np.isfinite(fallback)):
        raise ValueError("representations and fallback must be finite")
    count = seen.sum(axis=1)
    numerator = (rep * seen[..., None]).sum(axis=1)
    fused = np.divide(numerator, count[:, None], out=np.zeros_like(numerator), where=count[:, None] > 0)
    all_missing = count == 0
    fused[all_missing] = fallback
    diagnostics = FusionDiagnostics(count, count / rep.shape[1], all_missing)
    return fused, diagnostics


def sample_modality_keep_mask(
    available: np.ndarray,
    dropout_probability: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Drop available feeds without systematically favoring one modality.

    If dropout removes every available feed for a row, one of that row's
    genuinely available feeds is restored uniformly using the supplied,
    receipt-controlled random generator.
    """

    seen = np.asarray(available, dtype=bool)
    if seen.ndim != 2:
        raise ValueError("available must be [batch,modality]")
    if not np.isfinite(dropout_probability) or not 0 <= dropout_probability < 1:
        raise ValueError("dropout_probability must be in [0,1)")
    keep = seen & (rng.random(seen.shape) >= dropout_probability)
    for row in np.flatnonzero(seen.any(axis=1) & ~keep.any(axis=1)):
        candidates = np.flatnonzero(seen[row])
        keep[row, int(rng.choice(candidates))] = True
    return keep


@dataclass(frozen=True)
class BatchTargets:
    occurrence: np.ndarray
    occurrence_valid: np.ndarray
    peak_log_flux: np.ndarray
    peak_valid: np.ndarray
    onset_bin: np.ndarray
    onset_event: np.ndarray
    onset_valid: np.ndarray


@dataclass(frozen=True)
class LossResult:
    total: float
    per_task: dict[str, float]
    denominators: dict[str, int]
    active_tasks: tuple[str, ...]


def _masked_mean(loss: np.ndarray, valid: np.ndarray) -> tuple[float, int]:
    valid = np.asarray(valid, dtype=bool)
    denominator = int(valid.sum())
    return (float(loss[valid].mean()) if denominator else 0.0), denominator


def masked_binary_cross_entropy(logits: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[float, int]:
    logits = np.asarray(logits, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if logits.shape != target.shape or target.shape != valid.shape:
        raise ValueError("occurrence logits, targets, and validity mask must have equal shape")
    loss = np.maximum(logits, 0.0) - logits * target + np.log1p(np.exp(-np.abs(logits)))
    return _masked_mean(loss, valid)


def event_conditional_peak_flux_loss(
    prediction: np.ndarray,
    target_log_flux: np.ndarray,
    occurrence: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, int]:
    """Huber loss for peak log flux, evaluated only on valid positive events."""

    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target_log_flux, dtype=float)
    event_valid = np.asarray(occurrence, dtype=bool) & np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or target.shape != event_valid.shape:
        raise ValueError("peak prediction, target, and masks must have equal shape")
    error = prediction - target
    absolute = np.abs(error)
    loss = np.where(absolute <= 1.0, 0.5 * error**2, absolute - 0.5)
    return _masked_mean(loss, event_valid)


def discrete_time_censored_nll(
    hazard_logits: np.ndarray,
    onset_bin: np.ndarray,
    event_observed: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[float, int]:
    """Discrete hazard NLL with right censoring.

    ``onset_bin`` is zero-based for an event and equals ``K`` for a censored
    horizon of K bins.  A censored row contributes survival through all bins;
    an event contributes survival before its bin and the event hazard at it.
    """

    logits = np.asarray(hazard_logits, dtype=float)
    bins = np.asarray(onset_bin, dtype=int)
    events = np.asarray(event_observed, dtype=bool)
    usable = np.ones(len(bins), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if logits.ndim != 2 or bins.shape != (logits.shape[0],) or events.shape != bins.shape or usable.shape != bins.shape:
        raise ValueError("hazard logits must be [batch,bins] with matching target arrays")
    if not np.all(np.isfinite(logits)) or np.any(bins < 0) or np.any(bins > logits.shape[1]):
        raise ValueError("onset bins must lie in [0, number_of_bins]")
    if np.any(events & (bins == logits.shape[1])) or np.any((~events) & (bins != logits.shape[1])):
        raise ValueError("event rows need an onset bin; censored rows must use the K sentinel")
    losses = np.zeros(len(bins), dtype=float)
    for row, (row_logits, onset, event) in enumerate(zip(logits, bins, events)):
        survival_nll = np.logaddexp(0.0, row_logits)  # -log(1-hazard)
        if event:
            losses[row] = survival_nll[:onset].sum()
            losses[row] += np.logaddexp(0.0, -row_logits[onset])  # -log(hazard)
        else:
            losses[row] = survival_nll.sum()
    return _masked_mean(losses, usable)


def multitask_loss(predictions: Mapping[str, np.ndarray], targets: BatchTargets, config: ModelConfig) -> LossResult:
    """Compute the configured loss while never activating auxiliaries implicitly."""

    per_task: dict[str, float] = {}
    denominators: dict[str, int] = {}
    for task in config.active_tasks:
        if task == "occurrence":
            value, denominator = masked_binary_cross_entropy(
                predictions["occurrence_logits"], targets.occurrence, targets.occurrence_valid
            )
        elif task == "peak":
            value, denominator = event_conditional_peak_flux_loss(
                predictions["peak_prediction"], targets.peak_log_flux, targets.occurrence, targets.peak_valid
            )
        else:
            value, denominator = discrete_time_censored_nll(
                predictions["onset_logits"], targets.onset_bin, targets.onset_event, targets.onset_valid
            )
        per_task[task] = value
        denominators[task] = denominator
    total = float(sum(config.weight(task) * per_task[task] for task in config.active_tasks))
    return LossResult(total, per_task, denominators, config.active_tasks)


@dataclass(frozen=True)
class EnsembleSummary:
    probability: np.ndarray
    raw_mean: np.ndarray
    epistemic_std: np.ndarray
    lower_05: np.ndarray
    upper_95: np.ndarray
    abstain: np.ndarray


def ensemble_summary(
    probabilities: np.ndarray,
    *,
    all_missing: np.ndarray,
    fallback_probability: float,
) -> EnsembleSummary:
    """Return the uncertainty contract for a seed ensemble.

    ``epistemic_std`` is population spread across independently trained seeds;
    missing-input abstention is separate from model uncertainty.  Rows with no
    usable modality use the configured prior fallback probability and are
    explicitly marked ``abstain=True``.
    """

    values = np.asarray(probabilities, dtype=float)
    missing = np.asarray(all_missing, dtype=bool)
    if values.ndim != 2 or values.shape[1] != missing.shape[0] or values.shape[0] < 2:
        raise ValueError("probabilities must be [seeds,batch] with at least two seeds")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("ensemble probabilities must be finite and in [0,1]")
    if not np.isfinite(fallback_probability) or not 0 <= fallback_probability <= 1:
        raise ValueError("fallback_probability must be in [0,1]")
    raw_mean = values.mean(axis=0)
    probability = np.median(values, axis=0)
    probability[missing] = fallback_probability
    return EnsembleSummary(
        probability=probability,
        raw_mean=raw_mean,
        epistemic_std=values.std(axis=0),
        lower_05=np.quantile(values, 0.05, axis=0),
        upper_95=np.quantile(values, 0.95, axis=0),
        abstain=missing.copy(),
    )
