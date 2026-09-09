"""Constrained residual fusion for IRIS-SEP specialist probabilities.

System architecture:

    solar anchor logit
      + bounded, reliability-gated XRS residual
      + bounded, reliability-gated proton residual

The fusion layer deliberately operates on specialist probabilities rather than
raw heterogeneous features.  This preserves modality specialization and makes
it impossible for a softmax gate to suppress the solar anchor merely because a
context branch receives more representation mass.

This module is model-agnostic: the three specialist probabilities may come from
XGBoost, a compact neural model, or another frozen causal predictor.  The first
IRIS-SEP diagnostic uses the already-strong XGBoost specialists.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize


def _as_vector(name: str, value, *, length: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if length is not None and len(arr) != length:
        raise ValueError(f"{name} length mismatch")
    if len(arr) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _probability(name: str, value, *, length: int | None = None, eps: float) -> np.ndarray:
    arr = _as_vector(name, value, length=length)
    if ((arr < 0.0) | (arr > 1.0)).any():
        raise ValueError(f"{name} must lie in [0,1]")
    return np.clip(arr, eps, 1.0 - eps)


def _reliability(name: str, value, *, length: int) -> np.ndarray:
    if value is None:
        return np.ones(length, dtype=np.float64)
    arr = _as_vector(name, value, length=length)
    if ((arr < 0.0) | (arr > 1.0)).any():
        raise ValueError(f"{name} must lie in [0,1]")
    return arr


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1.0 + ez)
    return out


def _softplus(x: float) -> float:
    if x > 30.0:
        return float(x)
    return float(np.log1p(np.exp(x)))


def _inverse_softplus(y: float) -> float:
    if y <= 0:
        raise ValueError("softplus target must be positive")
    return float(np.log(np.expm1(y)))


@dataclass(frozen=True)
class ResidualFusionConfig:
    residual_logit_limit: float = 4.0
    l2_weight: float = 0.05
    initial_context_weight: float = 0.50
    probability_epsilon: float = 1e-6
    max_iter: int = 2000

    def __post_init__(self) -> None:
        if not math.isfinite(self.residual_logit_limit) or self.residual_logit_limit <= 0:
            raise ValueError("residual_logit_limit must be positive and finite")
        if not math.isfinite(self.l2_weight) or self.l2_weight < 0:
            raise ValueError("l2_weight must be finite and nonnegative")
        if not math.isfinite(self.initial_context_weight) or self.initial_context_weight <= 0:
            raise ValueError("initial_context_weight must be positive and finite")
        if not 0 < self.probability_epsilon < 0.5:
            raise ValueError("probability_epsilon must lie in (0,0.5)")
        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")


@dataclass(frozen=True)
class ResidualFusionFit:
    prevalence: float
    bias: float
    xrs_weight: float
    proton_weight: float
    optimizer_iterations: int
    optimizer_objective: float


class ResidualLogitFusion:
    """Tiny constrained meta-model over three specialist probabilities.

    The solar specialist is an immutable coefficient-1 anchor.  The two context
    specialists contribute only their bounded deviation from fit prevalence.
    Their amplitudes are nonnegative (softplus parameterization) and multiplied
    by externally supplied reliability values in [0,1].
    """

    def __init__(self, config: ResidualFusionConfig | None = None) -> None:
        self.config = config or ResidualFusionConfig()
        self.fit_: ResidualFusionFit | None = None

    def _features(
        self,
        solar_probability,
        xrs_probability,
        proton_probability,
        *,
        prevalence: float,
        xrs_reliability=None,
        proton_reliability=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eps = self.config.probability_epsilon
        solar = _probability("solar_probability", solar_probability, eps=eps)
        n = len(solar)
        xrs = _probability("xrs_probability", xrs_probability, length=n, eps=eps)
        proton = _probability("proton_probability", proton_probability, length=n, eps=eps)
        rx = _reliability("xrs_reliability", xrs_reliability, length=n)
        rp = _reliability("proton_reliability", proton_reliability, length=n)
        if not 0 < prevalence < 1:
            raise ValueError("prevalence must lie in (0,1)")
        climate = float(_logit(np.asarray([prevalence], dtype=np.float64))[0])
        limit = self.config.residual_logit_limit
        xrs_delta = np.clip(_logit(xrs) - climate, -limit, limit) * rx
        proton_delta = np.clip(_logit(proton) - climate, -limit, limit) * rp
        return _logit(solar), xrs_delta, proton_delta

    def fit(
        self,
        solar_probability,
        xrs_probability,
        proton_probability,
        y,
        *,
        xrs_reliability=None,
        proton_reliability=None,
    ) -> "ResidualLogitFusion":
        yv = _as_vector("y", y)
        if not np.isin(yv, [0.0, 1.0]).all() or len(np.unique(yv)) != 2:
            raise ValueError("y must contain both binary classes")
        prevalence = float(np.mean(yv))
        solar_z, xrs_delta, proton_delta = self._features(
            solar_probability,
            xrs_probability,
            proton_probability,
            prevalence=prevalence,
            xrs_reliability=xrs_reliability,
            proton_reliability=proton_reliability,
        )
        if len(solar_z) != len(yv):
            raise ValueError("probability and label length mismatch")

        raw0 = _inverse_softplus(self.config.initial_context_weight)
        x0 = np.asarray([0.0, raw0, raw0], dtype=np.float64)

        def objective(theta: np.ndarray) -> float:
            bias = float(theta[0])
            wx = _softplus(float(theta[1]))
            wp = _softplus(float(theta[2]))
            z = solar_z + bias + wx * xrs_delta + wp * proton_delta
            # Stable binary cross entropy in logit space.
            bce = np.mean(np.logaddexp(0.0, z) - yv * z)
            penalty = self.config.l2_weight * (bias * bias + wx * wx + wp * wp)
            return float(bce + penalty)

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": int(self.config.max_iter), "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"residual fusion optimization failed: {result.message}")
        self.fit_ = ResidualFusionFit(
            prevalence=prevalence,
            bias=float(result.x[0]),
            xrs_weight=_softplus(float(result.x[1])),
            proton_weight=_softplus(float(result.x[2])),
            optimizer_iterations=int(result.nit),
            optimizer_objective=float(result.fun),
        )
        return self

    def decision_function(
        self,
        solar_probability,
        xrs_probability,
        proton_probability,
        *,
        xrs_reliability=None,
        proton_reliability=None,
    ) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("fusion model is not fitted")
        solar_z, xrs_delta, proton_delta = self._features(
            solar_probability,
            xrs_probability,
            proton_probability,
            prevalence=self.fit_.prevalence,
            xrs_reliability=xrs_reliability,
            proton_reliability=proton_reliability,
        )
        return (
            solar_z
            + self.fit_.bias
            + self.fit_.xrs_weight * xrs_delta
            + self.fit_.proton_weight * proton_delta
        )

    def predict_proba(self, *args, **kwargs) -> np.ndarray:
        return _sigmoid(self.decision_function(*args, **kwargs))

    def diagnostics(self) -> dict[str, float | int]:
        if self.fit_ is None:
            raise RuntimeError("fusion model is not fitted")
        return {
            "fit_prevalence": self.fit_.prevalence,
            "bias": self.fit_.bias,
            "xrs_weight": self.fit_.xrs_weight,
            "proton_weight": self.fit_.proton_weight,
            "residual_logit_limit": self.config.residual_logit_limit,
            "l2_weight": self.config.l2_weight,
            "optimizer_iterations": self.fit_.optimizer_iterations,
            "optimizer_objective": self.fit_.optimizer_objective,
        }
