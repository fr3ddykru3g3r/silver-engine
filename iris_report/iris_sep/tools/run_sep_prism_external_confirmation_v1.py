from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

UPSTREAM_REPO = "yuyian/SEP-Prediction-V2"
UPSTREAM_COMMIT = "e138dcd72c1952a00e11e1a0b025337f9e7c93fb"
ROLLING_REL = Path("Data/rolling_combinded_seq_24hours.csv")
EVENT_REL = Path("GOES_integral_PRIMARY.1986-02-03.2025-09-10_sep_events.csv")
AGGREGATION_REL = Path("Rcode/Data_Aggregation.R")
EXPECTED_SHA256 = {
    str(ROLLING_REL): "4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb",
    str(EVENT_REL): "0ec9f0d6e088821091fcd369481bbbc9a2281a92fc8a582df40aefa62cae59b0",
}
EXPECTED_ROWS = 14464
EXPECTED_PUBLIC_POSITIVES = 650
START_COL = ">10.0 MeV 10.0 pfu SEP Start Time"
END_COL = ">10.0 MeV 10.0 pfu SEP End Time"

ERAS = [
    ("PRE_LOCAL_DEVELOPMENT_1986_2010", pd.Timestamp("1986-02-03T00:00:00Z"), pd.Timestamp("2011-01-01T00:00:00Z")),
    ("LOCAL_DEVELOPMENT_2011_2017", pd.Timestamp("2011-01-01T00:00:00Z"), pd.Timestamp("2018-01-01T00:00:00Z")),
    ("POST_LOCAL_DEVELOPMENT_2018_2025", pd.Timestamp("2018-01-01T00:00:00Z"), pd.Timestamp("2025-09-11T00:00:00Z")),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def build_episodes(events: pd.DataFrame) -> pd.DataFrame:
    if START_COL not in events.columns or END_COL not in events.columns:
        raise RuntimeError(f"event catalog missing required operational columns: {START_COL!r}, {END_COL!r}")
    starts = parse_time(events[START_COL])
    ends = parse_time(events[END_COL])
    complete = starts.notna() & ends.notna() & (ends > starts)
    rows: list[dict[str, Any]] = []
    for ordinal, idx in enumerate(events.index[complete], start=1):
        start = pd.Timestamp(starts.loc[idx])
        end = pd.Timestamp(ends.loc[idx])
        rows.append({
            "episode_id": f"PRISM_OP_{ordinal:05d}_{start.strftime('%Y%m%dT%H%M%SZ')}",
            "source_row_index": int(idx),
            "start": start,
            "end": end,
            "duration_hours": float((end - start).total_seconds() / 3600.0),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no complete operational >=10 pfu episodes")
    if out["episode_id"].duplicated().any():
        raise RuntimeError("duplicate generated episode id")
    return out.sort_values(["start", "end", "source_row_index"]).reset_index(drop=True)


def classify(issue: pd.Timestamp, horizon_end: pd.Timestamp, episodes: pd.DataFrame) -> dict[str, Any]:
    # Frozen external-confirmation semantics: onset endpoint inclusive.
    overlap = episodes[(episodes["start"] <= horizon_end) & (episodes["end"] >= issue)]
    active = episodes[(episodes["start"] <= issue) & (episodes["end"] > issue)]
    future = episodes[(episodes["start"] > issue) & (episodes["start"] <= horizon_end)]

    overlap_ids = overlap["episode_id"].astype(str).tolist()
    if len(active) > 1 or len(future) > 1:
        code = "DUPLICATE_OR_AMBIGUOUS"
        onset_id = None
    elif len(active) == 1:
        code = "ALREADY_ACTIVE_PERSISTENCE"
        onset_id = None
    elif len(future) == 1:
        code = "ELIGIBLE_ONSET_POSITIVE"
        onset_id = str(future.iloc[0]["episode_id"])
    else:
        code = "ELIGIBLE_ONSET_NEGATIVE"
        onset_id = None
    return {
        "interval_occurrence_label": int(bool(overlap_ids)),
        "eligibility_code": code,
        "onset_episode_id": onset_id,
        "occurrence_episode_match_count": int(len(overlap_ids)),
        "occurrence_episode_id_unique": overlap_ids[0] if len(overlap_ids) == 1 else None,
        "occurrence_episode_ids": ";".join(overlap_ids),
    }


def multiplicity(frame: pd.DataFrame, positive_col: str) -> dict[str, Any]:
    pos = frame[frame[positive_col] == 1].copy()
    unique = pos[pos["occurrence_episode_match_count"] == 1].copy()
    counts = (
        unique.groupby("occurrence_episode_id_unique", dropna=True)
        .size().rename("positive_window_count").reset_index()
    )
    n_ep = int(counts["occurrence_episode_id_unique"].nunique()) if len(counts) else 0
    n_win = int(len(unique))
    return {
        "positive_windows": int(len(pos)),
        "uniquely_mapped_positive_windows": n_win,
        "zero_episode_match_positive_windows": int((pos["occurrence_episode_match_count"] == 0).sum()),
        "multiple_episode_match_positive_windows": int((pos["occurrence_episode_match_count"] > 1).sum()),
        "represented_physical_episodes": n_ep,
        "episode_multiplicity_factor": float(n_win / n_ep) if n_ep else None,
        "median_positive_windows_per_episode": float(counts["positive_window_count"].median()) if len(counts) else None,
        "max_positive_windows_per_episode": int(counts["positive_window_count"].max()) if len(counts) else 0,
        "episodes_with_more_than_one_positive_window": int((counts["positive_window_count"] > 1).sum()) if len(counts) else 0,
        "episodes_with_at_least_three_positive_windows": int((counts["positive_window_count"] >= 3).sum()) if len(counts) else 0,
    }


def physical_state_summary(frame: pd.DataFrame) -> dict[str, Any]:
    reconstructed_pos = frame[frame["interval_occurrence_label"] == 1]
    persistence = int((reconstructed_pos["eligibility_code"] == "ALREADY_ACTIVE_PERSISTENCE").sum())
    onset = int((reconstructed_pos["eligibility_code"] == "ELIGIBLE_ONSET_POSITIVE").sum())
    ambiguous = int((reconstructed_pos["eligibility_code"] == "DUPLICATE_OR_AMBIGUOUS").sum())
    other = int(len(reconstructed_pos) - persistence - onset - ambiguous)
    return {
        "reconstructed_positive_windows": int(len(reconstructed_pos)),
        "already_active_persistence_windows": persistence,
        "new_onset_windows": onset,
        "ambiguous_positive_windows": ambiguous,
        "other_reconstructed_positive_windows": other,
        "persistence_to_onset_ratio": float(persistence / onset) if onset else None,
        "persistence_fraction_of_reconstructed_positives": float(persistence / len(reconstructed_pos)) if len(reconstructed_pos) else None,
    }


def era_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, start, end in ERAS:
        sub = frame[(frame["issue_timestamp"] >= start) & (frame["issue_timestamp"] < end)].copy()
        rec_mult = multiplicity(sub, "interval_occurrence_label")
        state = physical_state_summary(sub)
        out.append({
            "era": name,
            "start": start,
            "end_exclusive": end,
            "rows": int(len(sub)),
            "recorded_positive_windows": int((sub["recorded_operational_label"] == 1).sum()),
            "reconstructed_positive_windows": int((sub["interval_occurrence_label"] == 1).sum()),
            "uniquely_mapped_reconstructed_positive_windows": rec_mult["uniquely_mapped_positive_windows"],
            "represented_physical_episodes": rec_mult["represented_physical_episodes"],
            "episode_multiplicity_factor": rec_mult["episode_multiplicity_factor"],
            "persistence_windows": state["already_active_persistence_windows"],
            "new_onset_windows": state["new_onset_windows"],
            "target_mismatch_count": int((sub["recorded_operational_label"] != sub["interval_occurrence_label"]).sum()),
        })
    return out


def run(upstream: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise RuntimeError(f"output already exists: {out}")
    out.mkdir(parents=True)
    rolling_path = upstream / ROLLING_REL
    event_path = upstream / EVENT_REL
    aggregation_path = upstream / AGGREGATION_REL
    for path in [rolling_path, event_path, aggregation_path]:
        if not path.is_file():
            raise RuntimeError(f"missing pinned upstream file: {path}")

    source_manifest = []
    for rel, expected in EXPECTED_SHA256.items():
        path = upstream / rel
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"LFS SHA-256 mismatch for {rel}: {actual} != {expected}")
        source_manifest.append({"path": rel, "sha256": actual, "bytes": path.stat().st_size})
    source_manifest.append({
        "path": str(AGGREGATION_REL),
        "sha256": sha256_file(aggregation_path),
        "bytes": aggregation_path.stat().st_size,
    })

    rolling = pd.read_csv(rolling_path, low_memory=False)
    events = pd.read_csv(event_path, low_memory=False)
    required = {"window_begin", "window_end", "Future_OSEP_label"}
    missing = sorted(required - set(rolling.columns))
    if missing:
        raise RuntimeError(f"24-hour table missing required columns: {missing}")
    if len(rolling) != EXPECTED_ROWS:
        raise RuntimeError(f"unexpected 24-hour row count: {len(rolling)} != {EXPECTED_ROWS}")

    rolling["window_begin"] = parse_time(rolling["window_begin"])
    rolling["window_end"] = parse_time(rolling["window_end"])
    if rolling[["window_begin", "window_end"]].isna().any().any():
        raise RuntimeError("unparseable window timestamps")
    duration_h = (rolling["window_end"] - rolling["window_begin"]).dt.total_seconds() / 3600.0
    if not np.allclose(duration_h.to_numpy(float), 24.0, atol=1e-9, rtol=0):
        raise RuntimeError("24-hour table contains non-24-hour predictor windows")
    if rolling["window_end"].duplicated().any():
        raise RuntimeError("duplicate issue times in public 24-hour table")
    labels = pd.to_numeric(rolling["Future_OSEP_label"], errors="coerce")
    if labels.isna().any() or not set(labels.astype(int).unique()).issubset({0, 1}):
        raise RuntimeError("Future_OSEP_label is not complete binary data")
    rolling["Future_OSEP_label"] = labels.astype(int)
    if int(rolling["Future_OSEP_label"].sum()) != EXPECTED_PUBLIC_POSITIVES:
        raise RuntimeError(
            f"stored positive count differs from preregistered public metadata: {int(rolling['Future_OSEP_label'].sum())} != {EXPECTED_PUBLIC_POSITIVES}"
        )

    episodes = build_episodes(events)
    audit_rows: list[dict[str, Any]] = []
    for idx, row in rolling.iterrows():
        issue = pd.Timestamp(row["window_end"])
        end = issue + pd.Timedelta(hours=24)
        cls = classify(issue, end, episodes)
        audit_rows.append({
            "upstream_row_index": int(idx),
            "window_begin": row["window_begin"],
            "issue_timestamp": issue,
            "forecast_end": end,
            "recorded_operational_label": int(row["Future_OSEP_label"]),
            **cls,
        })
    frame = pd.DataFrame(audit_rows)
    frame["recorded_interval_match"] = frame["recorded_operational_label"] == frame["interval_occurrence_label"]

    # Episode-normalized physical occurrence weight: reconstructed positives only.
    reconstructed_unique = frame[(frame["interval_occurrence_label"] == 1) & (frame["occurrence_episode_match_count"] == 1)]
    counts = reconstructed_unique["occurrence_episode_id_unique"].value_counts().to_dict()
    weights = np.zeros(len(frame), dtype=float)
    for i, row in frame.iterrows():
        if int(row["interval_occurrence_label"]) == 0:
            weights[i] = 1.0
        elif int(row["occurrence_episode_match_count"]) == 1:
            weights[i] = 1.0 / counts[str(row["occurrence_episode_id_unique"])]
    frame["episode_normalized_physical_occurrence_weight"] = weights

    reconstructed_mult = multiplicity(frame, "interval_occurrence_label")
    recorded_mult = multiplicity(frame, "recorded_operational_label")
    state = physical_state_summary(frame)
    mismatch = int((~frame["recorded_interval_match"]).sum())
    ct = pd.crosstab(frame["recorded_operational_label"], frame["interval_occurrence_label"], dropna=False)
    cadence_h = frame["issue_timestamp"].sort_values().diff().dropna().dt.total_seconds() / 3600.0
    eras = era_summary(frame)

    emf = reconstructed_mult["episode_multiplicity_factor"]
    strong = bool(emf is not None and emf >= 1.5 and state["already_active_persistence_windows"] > state["new_onset_windows"])
    partial = bool((emf is not None and emf > 1.0) or state["already_active_persistence_windows"] > 0)
    disposition = "STRONG_CONFIRMATION" if strong else ("PARTIAL_CONFIRMATION" if partial else "NULL_OR_REFUTATION")

    summary = {
        "format": "IRIS_SEP_PRISM_EXTERNAL_CONFIRMATION_V1_RESULT",
        "status": disposition,
        "upstream": {
            "repository": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "source_manifest": source_manifest,
        },
        "window_table": {
            "rows": int(len(frame)),
            "stored_positive_windows": int(frame["recorded_operational_label"].sum()),
            "stored_negative_windows": int((frame["recorded_operational_label"] == 0).sum()),
            "predictor_window_hours": 24.0,
            "issue_cadence_hours_median": float(cadence_h.median()),
            "issue_cadence_hours_mode": float(cadence_h.mode().iloc[0]) if len(cadence_h) else None,
            "issue_cadence_24h_fraction": float(np.mean(np.isclose(cadence_h.to_numpy(float), 24.0))) if len(cadence_h) else None,
        },
        "physical_episode_catalog": {
            "complete_operational_episodes": int(len(episodes)),
            "median_duration_hours": float(episodes["duration_hours"].median()),
            "max_duration_hours": float(episodes["duration_hours"].max()),
        },
        "primary_reconstructed_multiplicity": reconstructed_mult,
        "stored_label_multiplicity_diagnostic": recorded_mult,
        "physical_state_decomposition": state,
        "target_reconciliation": {
            "mismatch_count": mismatch,
            "mismatch_fraction": float(mismatch / len(frame)),
            "crosstab": {
                str(int(r)): {str(int(c)): int(ct.loc[r, c]) for c in ct.columns}
                for r in ct.index
            },
        },
        "weights": {
            "ordinary_reconstructed_positive_window_mass": float(frame["interval_occurrence_label"].sum()),
            "episode_normalized_reconstructed_positive_mass": float(frame.loc[frame["interval_occurrence_label"] == 1, "episode_normalized_physical_occurrence_weight"].sum()),
            "expected_episode_normalized_mass": reconstructed_mult["represented_physical_episodes"],
        },
        "era_sensitivity": eras,
        "preregistered_gate": {
            "strong_confirmation_emf_at_least_1_5": bool(emf is not None and emf >= 1.5),
            "strong_confirmation_persistence_exceeds_onset": bool(state["already_active_persistence_windows"] > state["new_onset_windows"]),
            "disposition": disposition,
        },
        "secondary_model_analysis": {
            "performed": False,
            "reason": "primary model-free confirmation executed first; model analysis requires a separate admissibility check of timestamp-aligned predictions or leakage-safe public features"
        },
        "claim_boundary": {
            "external_public_confirmation": strong,
            "statistically_independent_from_all_prior_public_catalogs": False,
            "universal_literature_bias_claimed": False,
            "SEPNET_final_score_called_wrong": False,
            "protected_local_post_2025_outcomes_accessed": False,
        },
    }

    # Persist complete evidence.
    frame.to_csv(out / "sep_prism_window_audit.csv", index=False)
    episodes.to_csv(out / "sep_prism_episode_catalog_complete.csv", index=False)
    reconstructed_counts = (
        reconstructed_unique.groupby("occurrence_episode_id_unique").size()
        .rename("reconstructed_positive_window_count").reset_index()
        .sort_values(["reconstructed_positive_window_count", "occurrence_episode_id_unique"], ascending=[False, True])
    )
    reconstructed_counts.to_csv(out / "sep_prism_reconstructed_multiplicity_by_episode.csv", index=False)
    pd.DataFrame(eras).to_csv(out / "sep_prism_era_sensitivity.csv", index=False)
    ct.to_csv(out / "sep_prism_target_reconciliation_crosstab.csv")
    dump_json(out / "summary.json", summary)
    dump_json(out / "source_manifest.json", source_manifest)

    # Preserve exact upstream sources needed for independent reproduction.
    shutil.copy2(rolling_path, out / "upstream_rolling_combinded_seq_24hours.csv")
    shutil.copy2(event_path, out / "upstream_GOES_operational_event_catalog.csv")
    shutil.copy2(aggregation_path, out / "upstream_Data_Aggregation.R")

    files_to_hash = [
        "sep_prism_window_audit.csv",
        "sep_prism_episode_catalog_complete.csv",
        "sep_prism_reconstructed_multiplicity_by_episode.csv",
        "sep_prism_era_sensitivity.csv",
        "sep_prism_target_reconciliation_crosstab.csv",
        "summary.json",
        "source_manifest.json",
        "upstream_rolling_combinded_seq_24hours.csv",
        "upstream_GOES_operational_event_catalog.csv",
        "upstream_Data_Aggregation.R",
    ]
    evidence = {name: sha256_file(out / name) for name in files_to_hash}
    dump_json(out / "evidence_hashes.json", evidence)
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    summary = run(a.upstream_root, a.output)
    print(json.dumps({
        "status": summary["status"],
        "rows": summary["window_table"]["rows"],
        "physical_emf": summary["primary_reconstructed_multiplicity"]["episode_multiplicity_factor"],
        "persistence": summary["physical_state_decomposition"]["already_active_persistence_windows"],
        "onset": summary["physical_state_decomposition"]["new_onset_windows"],
        "mismatches": summary["target_reconciliation"]["mismatch_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
