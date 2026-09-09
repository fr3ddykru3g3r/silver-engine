"""Low-variance knowledge-distilled evidence stack for missing-feed IRIS-SEP states.

The student receives only the expert evidence that will remain available at
runtime.  During fit-only development it is trained against both the hard NEW-
SEP labels and a soft probability target from the full three-expert teacher.
The missing expert is never supplied to the student at inference time and no raw
measurement is reconstructed.
"""
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
class DistilledEvidenceStackConfig:
    expert_count: int
    teacher_weight: float = 0.35
    l2_weight: float = 0.03
    initial_weight: float = 0.50
    max_iter: int = 2000

    def __post_init__(self) -> None:
        if not isinstance(self.expert_count, int) or isinstance(self.expert_count, bool) or self.expert_count <= 0:
            raise ValueError("expert_count must be a positive integer")
        if not 0.0 <= self.teacher_weight < 1.0:
            raise ValueError("teacher_weight must be in [0,1)")
        if not math.isfinite(self.l2_weight) or self.l2_weight < 0:
            raise ValueError("l2_weight must be finite and nonnegative")
        if not math.isfinite(self.initial_weight) or self.initial_weight <= 0:
            raise ValueError("initial_weight must be finite and positive")
        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")


@dataclass(frozen=True)
class DistilledEvidenceStackFit:
    intercept: float
    weights: tuple[float, ...]
    optimizer_iterations: int
    optimizer_objective: float
    hard_bce: float
    teacher_bce: float


class DistilledPositiveEvidenceStack:
    """Nonnegative logistic student trained on hard and soft targets."""

    def __init__(self, config: DistilledEvidenceStackConfig) -> None:
        self.config = config
        self.fit_: DistilledEvidenceStackFit | None = None

    def _matrix(self, evidence) -> np.ndarray:
        x = np.asarray(evidence, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.config.expert_count or x.shape[0] == 0:
            raise ValueError(f"evidence must have shape [n,{self.config.expert_count}]")
        if not np.isfinite(x).all():
            raise ValueError("evidence must be finite")
        return x

    @staticmethod
    def _targets(y, teacher_probability, n: int) -> tuple[np.ndarray, np.ndarray]:
        hard = np.asarray(y, dtype=np.float64).reshape(-1)
        soft = np.asarray(teacher_probability, dtype=np.float64).reshape(-1)
        if len(hard) != n or len(soft) != n:
            raise ValueError("hard and teacher targets must align with evidence")
        if not np.isin(hard, [0.0, 1.0]).all() or len(np.unique(hard)) != 2:
            raise ValueError("hard labels must be binary and contain both classes")
        if not np.isfinite(soft).all() or ((soft < 0.0) | (soft > 1.0)).any():
            raise ValueError("teacher probabilities must be finite in [0,1]")
        return hard, soft

    def fit(self, evidence, y, teacher_probability) -> "DistilledPositiveEvidenceStack":
        x = self._matrix(evidence)
        hard, soft = self._targets(y, teacher_probability, len(x))
        raw0 = _inverse_softplus(self.config.initial_weight)
        theta0 = np.r_[0.0, np.full(self.config.expert_count, raw0, dtype=np.float64)]
        alpha = float(self.config.teacher_weight)

        def pieces(theta: np.ndarray) -> tuple[float, float, float]:
            intercept = float(theta[0])
            weights = np.asarray([_softplus(float(v)) for v in theta[1:]], dtype=np.float64)
            z = intercept + x @ weights
            log_norm = np.logaddexp(0.0, z)
            hard_bce = float(np.mean(log_norm - hard * z))
            teacher_bce = float(np.mean(log_norm - soft * z))
            penalty = self.config.l2_weight * float(np.sum(weights * weights))
            total = (1.0 - alpha) * hard_bce + alpha * teacher_bce + penalty
            return total, hard_bce, teacher_bce

        result = minimize(
            lambda theta: pieces(theta)[0],
            theta0,
            method="L-BFGS-B",
            options={"maxiter": self.config.max_iter, "ftol": 1e-12, "gtol": 1e-8},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"distilled evidence optimization failed: {result.message}")
        weights = tuple(_softplus(float(v)) for v in result.x[1:])
        total, hard_bce, teacher_bce = pieces(result.x)
        self.fit_ = DistilledEvidenceStackFit(
            intercept=float(result.x[0]),
            weights=weights,
            optimizer_iterations=int(result.nit),
            optimizer_objective=float(total),
            hard_bce=float(hard_bce),
            teacher_bce=float(teacher_bce),
        )
        return self

    def decision_function(self, evidence) -> np.ndarray:
        if self.fit_ is None:
            raise RuntimeError("distilled evidence stack is not fitted")
        x = self._matrix(evidence)
        return self.fit_.intercept + x @ np.asarray(self.fit_.weights, dtype=np.float64)

    def diagnostics(self) -> dict[str, object]:
        if self.fit_ is None:
            raise RuntimeError("distilled evidence stack is not fitted")
        return {
            "intercept": self.fit_.intercept,
            "weights": list(self.fit_.weights),
            "teacher_weight": self.config.teacher_weight,
            "hard_label_weight": 1.0 - self.config.teacher_weight,
            "l2_weight": self.config.l2_weight,
            "optimizer_iterations": self.fit_.optimizer_iterations,
            "optimizer_objective": self.fit_.optimizer_objective,
            "hard_bce": self.fit_.hard_bce,
            "teacher_bce": self.fit_.teacher_bce,
        }
