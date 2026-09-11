from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_ARCHIVE_SHA256 = "81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0"
EXPECTED = {
    "XGB_JOINT_MULTIPLICITY_EFFECT": {
        "median": -0.10374417093570104,
        "lo": -0.1461746092453961,
        "hi": -0.06316489217046042,
    },
    "XGB_JOINT_PERSISTENCE_REMOVAL_EFFECT": {
        "median": -0.18333333333333335,
        "lo": -0.24663865546218477,
        "hi": -0.12507002801120448,
    },
    "PROTON_HISTORY_PROXY_PERSISTENCE_EFFECT": {
        "median": -0.5061665245808469,
        "lo": -0.5666753044830267,
        "hi": -0.44397349240054246,
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metric_from_confusion(values: np.ndarray) -> np.ndarray:
    tp, fn, fp, tn = [values[..., i] for i in range(4)]
    with np.errstate(divide="ignore", invalid="ignore"):
        return tp / (tp + fn) - fp / (fp + tn)


def build_view(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "MAPPED_STANDARD":
        out = frame.copy()
        out["y"] = out["standard_label"].astype(int)
        out["weight"] = 1.0
        return out
    if name == "EPISODE_NORMALIZED_OCCURRENCE":
        out = frame.copy()
        out["y"] = out["standard_label"].astype(int)
        out["weight"] = 1.0
        counts = Counter(out.loc[out["y"].eq(1), "episode_id"].dropna().astype(str))
        mask = out["y"].eq(1)
        out.loc[mask, "weight"] = [1.0 / counts[str(x)] for x in out.loc[mask, "episode_id"]]
        return out
    if name == "NEW_ONSET":
        out = frame[frame["eligibility"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])].copy()
        out["y"] = out["eligibility"].eq("ELIGIBLE_ONSET_POSITIVE").astype(int)
        out["weight"] = 1.0
        return out
    raise ValueError(name)


def unit_contributions(
    view: pd.DataFrame,
    positive_units: np.ndarray,
    negative_units: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((len(positive_units), 4), dtype=float)
    neg = np.zeros((len(negative_units), 4), dtype=float)
    for i, unit in enumerate(positive_units.astype(str)):
        group = view[view["y"].eq(1) & view["episode_id"].astype(str).eq(unit)]
        if group.empty:
            raise RuntimeError(f"missing positive unit {unit}")
        w = group["weight"].to_numpy(float)
        a = group["alert"].to_numpy(int).astype(bool)
        pos[i, 0] = w[a].sum()
        pos[i, 1] = w[~a].sum()
    for i, unit in enumerate(negative_units.astype(str)):
        group = view[view["y"].eq(0) & view["quiet_block"].astype(str).eq(unit)]
        if group.empty:
            raise RuntimeError(f"missing negative unit {unit}")
        w = group["weight"].to_numpy(float)
        a = group["alert"].to_numpy(int).astype(bool)
        neg[i, 2] = w[a].sum()
        neg[i, 3] = w[~a].sum()
    return pos, neg


def summarize(delta: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.nanmedian(delta)),
        "lo": float(np.nanquantile(delta, 0.025)),
        "hi": float(np.nanquantile(delta, 0.975)),
        "pr_negative": float(np.nanmean(delta < 0)),
        "pr_positive": float(np.nanmean(delta > 0)),
    }


def point_tss(view: pd.DataFrame) -> float:
    y = view["y"].to_numpy(int)
    a = view["alert"].to_numpy(int).astype(bool)
    w = view["weight"].to_numpy(float)
    tp = w[(y == 1) & a].sum()
    fn = w[(y == 1) & ~a].sum()
    fp = w[(y == 0) & a].sum()
    tn = w[(y == 0) & ~a].sum()
    return float(tp / (tp + fn) - fp / (fp + tn))


def audit_dir(root: Path) -> dict[str, Any]:
    pred_path = root / "matched_episode_sensitivity_predictions.csv"
    draw_path = root / "shared_bootstrap_draws.npz"
    if not pred_path.exists() or not draw_path.exists():
        raise FileNotFoundError(
            "artifact must contain matched_episode_sensitivity_predictions.csv and shared_bootstrap_draws.npz"
        )

    predictions = pd.read_csv(pred_path, low_memory=False)
    draws = np.load(draw_path, allow_pickle=False)
    positive_units = draws["positive_units"].astype(str)
    negative_units = draws["negative_units"].astype(str)
    positive_indices = draws["positive_indices"]
    negative_indices = draws["negative_indices"]

    if positive_indices.shape != (10_000, len(positive_units)):
        raise RuntimeError("unexpected positive bootstrap tensor shape")
    if negative_indices.shape != (10_000, len(negative_units)):
        raise RuntimeError("unexpected negative bootstrap tensor shape")

    distributions: dict[tuple[str, str], np.ndarray] = {}
    points: dict[str, dict[str, float]] = {}
    for model in ["xgb_joint", "past_proton_ge10_proxy"]:
        frame = predictions[predictions["model"].eq(model)].copy()
        if frame.empty:
            raise RuntimeError(f"model missing: {model}")
        points[model] = {}
        for view_name in ["MAPPED_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET"]:
            view = build_view(frame, view_name)
            pos, neg = unit_contributions(view, positive_units, negative_units)
            sampled = pos[positive_indices].sum(axis=1) + neg[negative_indices].sum(axis=1)
            distributions[(model, view_name)] = metric_from_confusion(sampled)
            points[model][view_name] = point_tss(view)

    contrasts = {
        "XGB_JOINT_MULTIPLICITY_EFFECT": summarize(
            distributions[("xgb_joint", "EPISODE_NORMALIZED_OCCURRENCE")]
            - distributions[("xgb_joint", "MAPPED_STANDARD")]
        ),
        "XGB_JOINT_PERSISTENCE_REMOVAL_EFFECT": summarize(
            distributions[("xgb_joint", "NEW_ONSET")]
            - distributions[("xgb_joint", "EPISODE_NORMALIZED_OCCURRENCE")]
        ),
        "PROTON_HISTORY_PROXY_PERSISTENCE_EFFECT": summarize(
            distributions[("past_proton_ge10_proxy", "NEW_ONSET")]
            - distributions[("past_proton_ge10_proxy", "MAPPED_STANDARD")]
        ),
    }

    return {
        "matched_prediction_rows": int(len(predictions)),
        "rows_per_model": int(len(predictions) // predictions["model"].nunique()),
        "positive_episode_units": int(len(positive_units)),
        "negative_quiet_block_units": int(len(negative_units)),
        "point_tss": points,
        "primary_contrasts": contrasts,
    }


def assert_expected(result: dict[str, Any], atol: float = 1e-12) -> None:
    for key, expected in EXPECTED.items():
        actual = result["primary_contrasts"][key]
        for field, target in expected.items():
            if not np.isclose(actual[field], target, rtol=0.0, atol=atol):
                raise AssertionError(f"{key}.{field}: got {actual[field]!r}, expected {target!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Independent SEP-PRISM primary-contrast recomputation; imports no project runner code."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-zip", type=Path)
    source.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--assert-published", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    archive_sha = None
    if args.artifact_zip is not None:
        archive_sha = sha256_file(args.artifact_zip)
        with tempfile.TemporaryDirectory(prefix="sep_prism_recheck_") as td:
            with zipfile.ZipFile(args.artifact_zip) as zf:
                zf.extractall(td)
            result = audit_dir(Path(td))
    else:
        result = audit_dir(args.artifact_dir)

    if archive_sha is not None:
        result["artifact_zip_sha256"] = archive_sha
        result["published_artifact_sha256_match"] = archive_sha == EXPECTED_ARCHIVE_SHA256
        if args.assert_published and not result["published_artifact_sha256_match"]:
            raise AssertionError(f"archive SHA-256 mismatch: {archive_sha}")

    if args.assert_published:
        assert_expected(result)
        result["published_primary_contrasts_match"] = True

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
