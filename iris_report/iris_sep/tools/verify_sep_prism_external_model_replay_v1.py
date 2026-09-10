from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROLLING_SHA = "4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb"
EVENT_SHA = "0ec9f0d6e088821091fcd369481bbbc9a2281a92fc8a582df40aefa62cae59b0"
START_COL = ">10.0 MeV 10.0 pfu SEP Start Time"
END_COL = ">10.0 MeV 10.0 pfu SEP End Time"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc(values: Any) -> Any:
    return pd.to_datetime(values, utc=True, errors="coerce")


def episodes(events: pd.DataFrame) -> pd.DataFrame:
    starts = utc(events[START_COL])
    ends = utc(events[END_COL])
    ok = starts.notna() & ends.notna() & (ends > starts)
    rows = []
    for ordinal, idx in enumerate(events.index[ok], start=1):
        start = pd.Timestamp(starts.loc[idx])
        rows.append({
            "episode_id": f"PRISM_OP_{ordinal:05d}_{start.strftime('%Y%m%dT%H%M%SZ')}",
            "start": start,
            "end": pd.Timestamp(ends.loc[idx]),
        })
    return pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)


def classify(issue: pd.Timestamp, eps: pd.DataFrame) -> tuple[int, str, str | None, int]:
    horizon = issue + pd.Timedelta(hours=24)
    overlap = eps[(eps["start"] <= horizon) & (eps["end"] >= issue)]
    active = eps[(eps["start"] <= issue) & (eps["end"] > issue)]
    future = eps[(eps["start"] > issue) & (eps["start"] <= horizon)]
    ids = overlap["episode_id"].astype(str).tolist()
    if len(active) > 1 or len(future) > 1:
        code = "DUPLICATE_OR_AMBIGUOUS"
    elif len(active) == 1:
        code = "ALREADY_ACTIVE_PERSISTENCE"
    elif len(future) == 1:
        code = "ELIGIBLE_ONSET_POSITIVE"
    else:
        code = "ELIGIBLE_ONSET_NEGATIVE"
    return int(bool(ids)), code, (ids[0] if len(ids) == 1 else None), len(ids)


def metric(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y"].to_numpy(int)
    a = frame["alert"].to_numpy(int).astype(bool)
    p = frame["probability"].to_numpy(float)
    w = frame["weight"].to_numpy(float)
    tp = float(w[(y == 1) & a].sum())
    fn = float(w[(y == 1) & ~a].sum())
    fp = float(w[(y == 0) & a].sum())
    tn = float(w[(y == 0) & ~a].sum())
    pod = tp / (tp + fn) if tp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "pod": float(pod), "fpr": float(fpr), "far": float(far),
        "tss": float(pod - fpr),
        "hss": float(2 * (tp * tn - fn * fp) / denominator) if denominator else np.nan,
        "brier": float(np.sum(w * (p - y) ** 2) / w.sum()),
    }


def views(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    standard = frame.copy()
    standard["y"] = standard["standard_label"].astype(int)
    standard["weight"] = 1.0
    mapped = frame[(frame["standard_label"] == 0) | ((frame["standard_label"] == 1) & (frame["match_count"] == 1))].copy()
    mapped["y"] = mapped["standard_label"].astype(int)
    mapped["weight"] = 1.0
    normalized = mapped.copy()
    counts = Counter(normalized.loc[normalized["y"] == 1, "episode_id"].dropna().astype(str))
    normalized.loc[normalized["y"] == 1, "weight"] = [1.0 / counts[str(x)] for x in normalized.loc[normalized["y"] == 1, "episode_id"]]
    onset = frame[frame["eligibility"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"])].copy()
    onset["y"] = onset["eligibility"].eq("ELIGIBLE_ONSET_POSITIVE").astype(int)
    onset["weight"] = 1.0
    return {"STANDARD_FULL": standard, "MAPPED_STANDARD": mapped, "EPISODE_NORMALIZED_OCCURRENCE": normalized, "NEW_ONSET": onset}


def compare_metric_rows(recomputed: pd.DataFrame, reported: pd.DataFrame, keys: list[str], problems: list[str], label: str) -> None:
    metrics = ["tp", "fn", "fp", "tn", "pod", "fpr", "far", "tss", "hss", "brier"]
    joined = recomputed.merge(reported, on=keys, suffixes=("_r", "_p"), how="outer", indicator=True)
    if not joined["_merge"].eq("both").all():
        problems.append(f"{label} key mismatch")
        return
    for name in metrics:
        left = joined[f"{name}_r"].to_numpy(float)
        right = joined[f"{name}_p"].to_numpy(float)
        if not np.allclose(left, right, atol=1e-10, rtol=1e-9, equal_nan=True):
            problems.append(f"{label} metric mismatch: {name}")


def contributions(frame: pd.DataFrame, pos_units: list[str], neg_units: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((len(pos_units), 4), dtype=float)
    neg = np.zeros((len(neg_units), 4), dtype=float)
    for i, unit in enumerate(pos_units):
        g = frame[(frame["y"] == 1) & frame["episode_id"].eq(unit)]
        w = g["weight"].to_numpy(float)
        a = g["alert"].to_numpy(int).astype(bool)
        pos[i, 0], pos[i, 1] = w[a].sum(), w[~a].sum()
    for i, unit in enumerate(neg_units):
        g = frame[(frame["y"] == 0) & frame["quiet_block"].eq(unit)]
        w = g["weight"].to_numpy(float)
        a = g["alert"].to_numpy(int).astype(bool)
        neg[i, 2], neg[i, 3] = w[a].sum(), w[~a].sum()
    return pos, neg


def tss(confusion: np.ndarray) -> np.ndarray:
    tp, fn, fp, tn = [confusion[..., i] for i in range(4)]
    return tp / (tp + fn) - fp / (fp + tn)


def verify(root: Path) -> dict[str, Any]:
    problems: list[str] = []
    rolling_path = root / "upstream_rolling_combinded_seq_24hours.csv"
    event_path = root / "upstream_GOES_operational_event_catalog.csv"
    if sha(rolling_path) != ROLLING_SHA:
        problems.append("rolling source sha mismatch")
    if sha(event_path) != EVENT_SHA:
        problems.append("event source sha mismatch")

    evidence = json.loads((root / "evidence_hashes.json").read_text())
    bad = [name for name, expected in evidence.items() if not (root / name).is_file() or sha(root / name) != expected]
    if bad:
        problems.append("evidence hash mismatch: " + ",".join(bad))

    rolling = pd.read_csv(rolling_path, low_memory=False)
    rolling["window_end"] = utc(rolling["window_end"])
    events = pd.read_csv(event_path, low_memory=False)
    eps = episodes(events)
    schema = json.loads((root / "feature_schema.json").read_text())
    excluded = {"window_begin", "window_end", "OSEP_label", "GSEP_label"}
    expected_joint = [c for c in rolling.columns if c not in excluded and not c.startswith("Future_")]
    expected_no_proton = [c for c in expected_joint if c != "ProtonFlux_label" and not c.startswith("ProtonFlux_")]
    if schema["joint"] != expected_joint or schema["no_proton"] != expected_no_proton:
        problems.append("feature schema does not match frozen rule")

    pred = pd.read_csv(root / "predictions.csv")
    pred["issue_timestamp"] = utc(pred["issue_timestamp"])
    if pred.duplicated(["fold", "issue_timestamp", "model"]).any():
        problems.append("duplicate prediction rows")
    if ((pred["probability"] < 0) | (pred["probability"] > 1) | ~np.isfinite(pred["probability"])).any():
        problems.append("invalid prediction probability")
    if not np.array_equal(pred["alert"].to_numpy(int), (pred["probability"] >= pred["threshold"]).astype(int).to_numpy()):
        problems.append("alerts do not match frozen row thresholds")
    signatures = []
    for _, group in pred.groupby("model"):
        signatures.append(tuple(zip(group["fold"].astype(str), group["issue_timestamp"].astype(str))))
    if len(set(signatures)) != 1:
        problems.append("fixed models do not share identical score cohorts")

    base_model = sorted(pred["model"].unique())[0]
    base = pred[pred["model"].eq(base_model)].copy()
    row_checks = []
    for _, row in base.iterrows():
        y, code, episode_id, match_count = classify(pd.Timestamp(row["issue_timestamp"]), eps)
        row_checks.append((y, code, episode_id, match_count))
    if not np.array_equal(base["standard_label"].to_numpy(int), np.array([x[0] for x in row_checks], dtype=int)):
        problems.append("prediction standard labels do not reproduce from event catalog")
    if base["eligibility"].astype(str).tolist() != [x[1] for x in row_checks]:
        problems.append("prediction eligibility does not reproduce from event catalog")
    if not np.array_equal(base["match_count"].to_numpy(int), np.array([x[3] for x in row_checks], dtype=int)):
        problems.append("prediction episode match counts do not reproduce")
    persisted_ids = base["episode_id"].where(base["episode_id"].notna(), None).tolist()
    if persisted_ids != [x[2] for x in row_checks]:
        problems.append("prediction unique episode IDs do not reproduce")

    desc_rows = []
    for model in sorted(pred["model"].unique()):
        for view_name, frame in views(pred[pred["model"].eq(model)]).items():
            desc_rows.append({"model": model, "view": view_name, **metric(frame)})
    compare_metric_rows(pd.DataFrame(desc_rows), pd.read_csv(root / "descriptive_results.csv"), ["model", "view"], problems, "descriptive")

    temporal_rows = []
    for fold in sorted(pred["fold"].unique()):
        for model in sorted(pred["model"].unique()):
            model_views = views(pred[pred["fold"].eq(fold) & pred["model"].eq(model)])
            for view_name in ["STANDARD_FULL", "NEW_ONSET"]:
                temporal_rows.append({"fold": fold, "model": model, "view": view_name, **metric(model_views[view_name])})
    compare_metric_rows(pd.DataFrame(temporal_rows), pd.read_csv(root / "per_fold_results.csv"), ["fold", "model", "view"], problems, "temporal")

    draws = np.load(root / "shared_bootstrap_draws.npz", allow_pickle=False)
    pi, ni = draws["positive_indices"], draws["negative_indices"]
    pos_units = draws["positive_units"].astype(str).tolist()
    neg_units = draws["negative_units"].astype(str).tolist()
    expected_pos = sorted(set(base.loc[base["eligibility"].eq("ELIGIBLE_ONSET_POSITIVE") & base["match_count"].eq(1), "episode_id"].dropna().astype(str)))
    expected_neg = sorted(set(base.loc[base["standard_label"].eq(0), "quiet_block"].astype(str)))
    if pos_units != expected_pos or neg_units != expected_neg:
        problems.append("bootstrap physical-unit population mismatch")
    if pi.shape != (10000, len(pos_units)) or ni.shape != (10000, len(neg_units)):
        problems.append("bootstrap draw tensor shape mismatch")

    sensitivity = pred[pred["standard_label"].eq(0) | (pred["match_count"].eq(1) & pred["episode_id"].astype(str).isin(pos_units))].copy()
    saved_sensitivity = pd.read_csv(root / "matched_episode_sensitivity_predictions.csv")
    if len(sensitivity) != len(saved_sensitivity):
        problems.append("matched sensitivity row count mismatch")

    distributions: dict[tuple[str, str], np.ndarray] = {}
    points = []
    for model in sorted(pred["model"].unique()):
        model_views = views(sensitivity[sensitivity["model"].eq(model)])
        for view_name in ["MAPPED_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET"]:
            frame = model_views[view_name]
            points.append({"model": model, "view": view_name, **metric(frame)})
            cp, cn = contributions(frame, pos_units, neg_units)
            distributions[(model, view_name)] = tss(cp[pi].sum(axis=1) + cn[ni].sum(axis=1))
    compare_metric_rows(pd.DataFrame(points), pd.read_csv(root / "matched_episode_point_results.csv"), ["model", "view"], problems, "matched-point")

    def contrast(model: str, earlier: str, later: str) -> dict[str, float]:
        delta = distributions[(model, later)] - distributions[(model, earlier)]
        return {
            "median": float(np.nanmedian(delta)), "lo": float(np.nanquantile(delta, 0.025)),
            "hi": float(np.nanquantile(delta, 0.975)), "pr_negative": float(np.nanmean(delta < 0)),
            "pr_positive": float(np.nanmean(delta > 0)),
        }
    recalculated: dict[str, Any] = {
        "XGB_JOINT_MULTIPLICITY_EFFECT": contrast("xgb_joint", "MAPPED_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE"),
        "XGB_JOINT_PERSISTENCE_REMOVAL_EFFECT": contrast("xgb_joint", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET"),
        "PROTON_HISTORY_PROXY_PERSISTENCE_EFFECT": contrast("past_proton_ge10_proxy", "MAPPED_STANDARD", "NEW_ONSET"),
    }
    for family, joint, no_proton in [("XGB", "xgb_joint", "xgb_no_proton"), ("ELASTIC", "elastic_net_joint", "elastic_net_no_proton")]:
        onset_delta = distributions[(joint, "NEW_ONSET")] - distributions[(no_proton, "NEW_ONSET")]
        standard_delta = distributions[(joint, "MAPPED_STANDARD")] - distributions[(no_proton, "MAPPED_STANDARD")]
        recalculated[f"{family}_JOINT_MINUS_NO_PROTON_ONSET"] = {
            "median": float(np.nanmedian(onset_delta)), "lo": float(np.nanquantile(onset_delta, 0.025)),
            "hi": float(np.nanquantile(onset_delta, 0.975)), "pr_positive": float(np.nanmean(onset_delta > 0)),
            "ordering_reversal_fraction": float(np.nanmean(np.sign(standard_delta) != np.sign(onset_delta))),
        }
    reported = json.loads((root / "bootstrap_contrasts.json").read_text())
    for key, values in recalculated.items():
        for field, value in values.items():
            if not np.isclose(float(value), float(reported[key][field]), atol=1e-12, rtol=1e-10, equal_nan=True):
                problems.append(f"bootstrap contrast mismatch {key}/{field}")

    directional = [recalculated[k] for k in ["XGB_JOINT_MULTIPLICITY_EFFECT", "XGB_JOINT_PERSISTENCE_REMOVAL_EFFECT", "PROTON_HISTORY_PROXY_PERSISTENCE_EFFECT"]]
    supported = sum(item["hi"] < 0 for item in directional)
    required = recalculated["XGB_JOINT_PERSISTENCE_REMOVAL_EFFECT"]["hi"] < 0 or recalculated["PROTON_HISTORY_PROXY_PERSISTENCE_EFFECT"]["hi"] < 0
    gate = "STRONG_MODEL_CONFIRMATION" if supported >= 2 and required else ("PARTIAL_MODEL_CONFIRMATION" if supported >= 1 else "NULL_MODEL_CONFIRMATION")
    summary = json.loads((root / "summary.json").read_text())
    if summary["status"] != gate:
        problems.append(f"model confirmation gate mismatch: {summary['status']} != {gate}")

    return {
        "format": "IRIS_SEP_PRISM_EXTERNAL_FIXED_MODEL_REPLAY_INDEPENDENT_VERIFICATION_V1",
        "passed": not problems,
        "problems": problems,
        "recomputed_gate": gate,
        "score_rows_unique": int(base["issue_timestamp"].nunique()),
        "models": sorted(pred["model"].unique()),
        "matched_onset_episode_units": len(pos_units),
        "quiet_block_units": len(neg_units),
        "recomputed_primary_contrasts": recalculated,
        "source_sha256": {"rolling": sha(rolling_path), "events": sha(event_path)},
        "evidence_files_verified": len(evidence),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.artifact_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
