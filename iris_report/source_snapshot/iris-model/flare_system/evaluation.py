from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from metrics import all_metrics, region_bootstrap


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def atomic_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False, default=str) + "\n")
    os.replace(temporary, destination)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_tss_threshold(frame: pd.DataFrame) -> tuple[float, dict]:
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 197), frame.p.to_numpy(float)])
    best: tuple[tuple[float, float, float], float, dict] | None = None
    prevalence = float(frame.y.mean())
    for threshold in candidates:
        metrics = all_metrics(frame.y, frame.p, float(threshold))
        score = metrics["tss"] if np.isfinite(metrics["tss"]) else -999.0
        key = (score, metrics.get("hss", -999.0), -abs(float(threshold) - prevalence))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return best[1], best[2]


class LogitCalibrator:
    def __init__(self) -> None:
        self.temperature = 1.0
        self.bias = 0.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        x = torch.tensor(logits, dtype=torch.float64)
        y = torch.tensor(labels, dtype=torch.float64)
        log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
        bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature, bias], lr=0.2, max_iter=80, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            temperature = log_temperature.exp().clamp(0.05, 20.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(x / temperature + bias, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
        self.bias = float(bias.detach())
        return self.temperature

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float64) / self.temperature + self.bias
        scaled = np.clip(scaled, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-scaled))


def ensemble_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No prediction frames")
    keys = ["sample_id", "region_group_id", "y"]
    merged = frames[0][keys + ["p"]].rename(columns={"p": "p_0"})
    for index, frame in enumerate(frames[1:], 1):
        merged = merged.merge(
            frame[keys + ["p"]].rename(columns={"p": f"p_{index}"}),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    if len(merged) != len(frames[0]) or any(len(frame) != len(merged) for frame in frames):
        raise RuntimeError("Seed predictions do not contain identical frozen identities")
    columns = [column for column in merged if column.startswith("p_")]
    merged["p"] = merged[columns].mean(axis=1)
    merged["p_std"] = merged[columns].std(axis=1, ddof=0)
    return merged


def ensemble_logits(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No prediction frames")
    keys = ["sample_id", "region_group_id", "y"]
    merged = frames[0][keys + ["logit"]].rename(columns={"logit": "logit_0"})
    for index, frame in enumerate(frames[1:], 1):
        merged = merged.merge(
            frame[keys + ["logit"]].rename(columns={"logit": f"logit_{index}"}),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    if len(merged) != len(frames[0]) or any(len(frame) != len(merged) for frame in frames):
        raise RuntimeError("Seed logits do not contain identical frozen identities")
    columns = [column for column in merged if column.startswith("logit_")]
    values = merged[columns].to_numpy(float)
    merged["logit"] = values.mean(axis=1)
    raw_probabilities = 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))
    merged["p_std"] = raw_probabilities.std(axis=1, ddof=0)
    return merged


def reliability_table(frame: pd.DataFrame, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (frame.p >= low) & (frame.p < high if index + 1 < bins else frame.p <= high)
        subset = frame[mask]
        rows.append(
            {
                "bin": index,
                "low": float(low),
                "high": float(high),
                "count": int(len(subset)),
                "mean_probability": float(subset.p.mean()) if len(subset) else None,
                "event_rate": float(subset.y.mean()) if len(subset) else None,
            }
        )
    return rows


def physics_baseline(
    train_features: np.ndarray,
    train_y: np.ndarray,
    calibration_features: np.ndarray,
    calibration_y: np.ndarray,
    threshold_features: np.ndarray,
    threshold_y: np.ndarray,
    test_features: np.ndarray,
    test_frame: pd.DataFrame,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(train_features, train_y)
    calibrator = LogitCalibrator()
    calibrator.fit(model.decision_function(calibration_features), calibration_y)
    validation = pd.DataFrame(
        {"y": threshold_y, "p": calibrator.probabilities(model.decision_function(threshold_features))}
    )
    threshold, validation_metrics = choose_tss_threshold(validation)
    test = test_frame[["sample_id", "region_group_id", "y"]].copy()
    test["p"] = calibrator.probabilities(model.decision_function(test_features))
    return {
        "threshold": threshold,
        "validation": validation_metrics,
        "test": all_metrics(test.y, test.p, threshold),
        "test_region_bootstrap": region_bootstrap(test, n_boot=2000, seed=seed + 91, threshold=threshold),
        "calibrator": {"temperature": calibrator.temperature, "bias": calibrator.bias},
        "coefficients": model.coef_[0].tolist(),
        "intercept": model.intercept_.tolist(),
    }, test


def paired_region_bootstrap_delta(
    frame: pd.DataFrame,
    model_threshold: float,
    baseline_threshold: float,
    n_boot: int,
    seed: int,
) -> dict:
    """Paired connected-region bootstrap for model-minus-baseline metrics."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(sorted(frame.region_group_id.astype(str).unique()))
    values = {name: [] for name in ("tss", "hss", "auroc", "auprc", "brier", "bss")}
    for _ in range(n_boot):
        draw = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat(
            [frame[frame.region_group_id.astype(str).eq(group)] for group in draw],
            ignore_index=True,
        )
        model_metrics = all_metrics(sample.y, sample.p, model_threshold)
        baseline_metrics = all_metrics(sample.y, sample.p_baseline, baseline_threshold)
        for name in values:
            values[name].append(float(model_metrics[name] - baseline_metrics[name]))
    return {
        name: {
            "median": float(np.nanmedian(delta)),
            "lo95": float(np.nanpercentile(delta, 2.5)),
            "hi95": float(np.nanpercentile(delta, 97.5)),
        }
        for name, delta in values.items()
    }
