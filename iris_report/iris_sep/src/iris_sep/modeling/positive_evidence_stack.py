"""Small nonnegative meta-model for cross-fitted IRIS-SEP specialist evidence."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize


def _softplus(x: float) -> float:
    if x > 30.0:
        return float(x)
    return float(np.log1p(np.exp(x)))


def _inverse_softplus(y: float) -> float:
    if y <= 0:
        raise ValueError("initial weight must be positive")
    return float(np.log(np.expm1(y)))


@dataclass(frozen=True)
class EvidenceStackConfig:
    expert_count: int = 3
    l2_weight: float = 0.02
    initial_weight: float = 0.50
    max_iter: int = 2000

    def __post_init__(self) -> None:
        if not isinstance(self.expert_count, int) or isinstance(self.expert_count, bool) or self.expert_count <= 0:
            raise ValueError("expert_count must be a positive integer")
        if not math.isfinite(self.l2_weight) or self.l2_weight < 0:
            raise ValueError("l2_weight must be finite and nonnegative")
        if not math.isfinite(self.initial_weight) or self.initial_weight <= 0:
            raise ValueError("initial_weight must be finite and positive")
        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")


@dataclass(frozen=True)
class EvidenceStackFit:
    intercept: float
    weights: tuple[float, ...]
    optimizer_iterations: int
    optimizer_objective: float


class PositiveEvidenceStack:
    """Logistic stack whose specialist evidence coefficients cannot be negative."""

    def __init__(self, config: EvidenceStackConfig | None = None) -> None:
        self.config = config or EvidenceStackConfig()
        self.fit_: EvidenceStackFit | None = None

    def _matrix(self, evidence) -> np.ndarray:
        x = np.asarray(evidence, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.config.expert_count or x.shape[0] == 0:
            raise ValueError(f"evidence must have shape [n,{self.config.expert_count}]")
        if not np.isfinite(x).all():
            raise ValueError("evidence must be finite")
        return x

    def fit(self, evidence, y) -> "PositiveEvidenceStack":
        x = self._matrix(evidence)
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(yy) != len(x) or not np.isin(yy, [0.0, 1.0]).all() or len(np.unique(yy)) != 2:
            raise ValueError("y must be aligned binary labels containing both classes")
        raw0 = _inverse_softplus(self.config.initial_weight)
        theta0 = np.r_[0.0, np.full(self.config.expert_count, raw0, dtype=np.float64)]

        def objective(theta: np.ndarray) -> float:
            intercept = float(theta[0])
            weights = np.asarray([_softplus(float(v)) for v in theta[1:]], dtype=np.float64)
            z = intercept + x @ weights
            bce = float(np.mean(np.logaddexp(0.0, z) - yy * z))
            penalty = self.config.l2_weight * float(np.sum(weights * weights))
            return bce + penalty

        result = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": self.config.max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"evidence stack optimization failed: {result.message}")
        weights = tuple(_softplus(float(v)) for v in result.x[1:])
        self.fit_ = EvidenceStackFit(
            intercept=float(result.x[0]),
            weights=weights,
            optimizer_iterations=int(result.nit),
            optimizer_objective=float(result.fun),
        )
        return self

    def decision_function(self, evidence) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("evidence stack is not fitted")
        x = self._matrix(evidence)
        return self.fit_.intercept + x @ np.asarray(self.fit_.weights, dtype=np.float64)

    def diagnostics(self) -> dict[str, object]:
        if self.fit_ is None:
            raise RuntimeError("evidence stack is not fitted")
        return {
            "intercept": self.fit_.intercept,
            "weights": list(self.fit_.weights),
            "l2_weight": self.config.l2_weight,
            "optimizer_iterations": self.fit_.optimizer_iterations,
            "optimizer_objective": self.fit_.optimizer_objective,
        }
