from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import requests

from tools.episode_semantics import (
    OperationalEpisode,
    attrition_counts,
    classify_window_from_episode_intervals,
    episode_normalized_occurrence_weights,
)

UPSTREAM_REPO = "yuyian/SEP-Prediction"
UPSTREAM_COMMIT = "d0eb54e46b7dd6c760325e123d2ad86f9420fbff"
RAW_ROOT = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{UPSTREAM_COMMIT}"
FILES = {
    "rolling": "data/rolling_combinded.csv",
    "events": "data/df_SEP.csv",
    "construction": "Data-Construct.R",
    "readme": "README.md",
}
EXPECTED_GIT_BLOBS = {
    "rolling": "82cf23dc789e70d8185dd6c69bb6b3bd7692f77b",
    "events": "dc1ea7a18c2b06034a144f656d76f4dcd5e85674",
    "construction": "8234c4e8e7814b1d0cc477263e82504b1901127a",
    "readme": "810f329f9120a6020445f4c5a62be219206bc4d6",
}
UA = "IRIS-SEP-episode-benchmark-published-audit/1.0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def fetch(session: requests.Session, url: str, *, maximum: int = 20_000_000) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(url, timeout=90, headers={"User-Agent": UA})
            response.raise_for_status()
            body = response.content
            if not body:
                raise RuntimeError(f"empty response: {url}")
            if len(body) > maximum:
                raise RuntimeError(f"response exceeds bound: {url}: {len(body)}")
            return body
        except Exception as exc:  # bounded network retry only
            last = exc
            if attempt == 4:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"failed after bounded retries: {url}") from last


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def load_sources(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    session = requests.Session()
    bodies: dict[str, bytes] = {}
    manifest: list[dict[str, Any]] = []
    for key, relative in FILES.items():
        url = f"{RAW_ROOT}/{relative}"
        body = fetch(session, url)
        bodies[key] = body
        blob_sha = git_blob_sha1(body)
        if blob_sha != EXPECTED_GIT_BLOBS[key]:
            raise RuntimeError(f"upstream Git blob mismatch for {relative}: {blob_sha}")
        manifest.append(
            {
                "key": key,
                "repository": UPSTREAM_REPO,
                "commit": UPSTREAM_COMMIT,
                "path": relative,
                "url": url,
                "bytes": len(body),
                "git_blob_sha1": blob_sha,
                "sha256": sha256_bytes(body),
            }
        )
    (out / "upstream_Data-Construct.R").write_bytes(bodies["construction"])
    (out / "upstream_README.md").write_bytes(bodies["readme"])
    rolling_path = out / "upstream_rolling_combinded.csv"
    events_path = out / "upstream_df_SEP.csv"
    rolling_path.write_bytes(bodies["rolling"])
    events_path.write_bytes(bodies["events"])
    return pd.read_csv(rolling_path), pd.read_csv(events_path), manifest


def build_operational_episodes(events: pd.DataFrame) -> tuple[list[OperationalEpisode], pd.DataFrame]:
    start_col = "start_time" if "start_time" in events else ">10.0 MeV 10.0 pfu SEP Start Time"
    end_col = "end_time" if "end_time" in events else ">10.0 MeV 10.0 pfu SEP End Time"
    starts = parse_time(events[start_col])
    ends = parse_time(events[end_col])
    complete = starts.notna() & ends.notna() & (ends > starts)
    rows: list[dict[str, Any]] = []
    episodes: list[OperationalEpisode] = []
    for ordinal, idx in enumerate(events.index[complete], start=1):
        start = pd.Timestamp(starts.loc[idx])
        end = pd.Timestamp(ends.loc[idx])
        episode_id = f"SEPNET_OP_{ordinal:05d}_{start.strftime('%Y%m%dT%H%M%SZ')}"
        episodes.append(OperationalEpisode(episode_id, start, end))
        rows.append(
            {
                "episode_id": episode_id,
                "source_row_index": int(idx),
                "start": start,
                "end": end,
                "duration_hours": (end - start).total_seconds() / 3600.0,
            }
        )
    return episodes, pd.DataFrame(rows)


def audit(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=False)
    rolling, events, manifest = load_sources(out)
    required = {"window_begin", "window_end", "future_Operational_SEP_label"}
    missing = sorted(required - set(rolling.columns))
    if missing:
        raise RuntimeError(f"rolling table missing required columns: {missing}")

    rolling["window_begin"] = parse_time(rolling["window_begin"])
    rolling["window_end"] = parse_time(rolling["window_end"])
    if rolling[["window_begin", "window_end"]].isna().any().any():
        raise RuntimeError("rolling table contains unparseable window timestamps")
    if not (rolling["window_end"] > rolling["window_begin"]).all():
        raise RuntimeError("non-positive predictor window duration")
    duration_hours = (rolling["window_end"] - rolling["window_begin"]).dt.total_seconds() / 3600
    if not np.allclose(duration_hours.to_numpy(float), 24.0):
        raise RuntimeError("published rolling table is not uniformly 24-hour windows")
    if not set(pd.unique(rolling["future_Operational_SEP_label"].dropna())).issubset({0, 1}):
        raise RuntimeError("operational target is not binary")

    episodes, episode_table = build_operational_episodes(events)
    if not episodes:
        raise RuntimeError("no complete operational threshold episodes in upstream event table")

    audit_rows: list[dict[str, Any]] = []
    for idx, row in rolling.iterrows():
        issue = pd.Timestamp(row["window_end"])
        horizon_end = issue + pd.Timedelta(hours=24)
        recorded = int(row["future_Operational_SEP_label"])
        cls = classify_window_from_episode_intervals(
            issue,
            horizon_end,
            episodes,
            recorded_standard_label=recorded,
        )
        unique_occurrence_id = cls.occurrence_episode_ids[0] if len(cls.occurrence_episode_ids) == 1 else None
        audit_rows.append(
            {
                "upstream_row_index": int(idx),
                "window_begin": row["window_begin"],
                "issue_timestamp": issue,
                "forecast_end": horizon_end,
                "recorded_operational_label": recorded,
                "interval_occurrence_label": cls.interval_occurrence_label,
                "recorded_interval_match": cls.standard_label_matches_intervals,
                "eligibility_code": cls.eligibility_code,
                "onset_episode_id": cls.onset_episode_id,
                "occurrence_episode_match_count": len(cls.occurrence_episode_ids),
                "occurrence_episode_id_unique": unique_occurrence_id,
                "occurrence_episode_ids": ";".join(cls.occurrence_episode_ids),
            }
        )
    frame = pd.DataFrame(audit_rows)

    occurrence_weights = episode_normalized_occurrence_weights(
        frame["occurrence_episode_id_unique"].where(frame["occurrence_episode_id_unique"].notna(), None).tolist(),
        frame["recorded_operational_label"].astype(int).tolist(),
    )
    frame["episode_normalized_occurrence_weight"] = occurrence_weights
    frame["standard_window_weight"] = 1.0
    frame["onset_weight"] = np.where(
        frame["eligibility_code"].isin(["ELIGIBLE_ONSET_POSITIVE", "ELIGIBLE_ONSET_NEGATIVE"]),
        1.0,
        0.0,
    )

    positive = frame[frame["recorded_operational_label"] == 1].copy()
    uniquely_mapped = positive[positive["occurrence_episode_match_count"] == 1].copy()
    per_episode = (
        uniquely_mapped.groupby("occurrence_episode_id_unique", dropna=True)
        .size()
        .rename("positive_window_count")
        .reset_index()
        .sort_values(["positive_window_count", "occurrence_episode_id_unique"], ascending=[False, True])
    )
    episode_table = episode_table.merge(
        per_episode,
        how="left",
        left_on="episode_id",
        right_on="occurrence_episode_id_unique",
    ).drop(columns=["occurrence_episode_id_unique"], errors="ignore")
    episode_table["positive_window_count"] = episode_table["positive_window_count"].fillna(0).astype(int)

    mapped_distinct = int(uniquely_mapped["occurrence_episode_id_unique"].nunique())
    emf = float(len(uniquely_mapped) / mapped_distinct) if mapped_distinct else float("nan")
    mismatch = frame[frame["recorded_interval_match"] == False]  # noqa: E712
    label_crosstab = pd.crosstab(
        frame["recorded_operational_label"],
        frame["interval_occurrence_label"],
        rownames=["recorded"],
        colnames=["episode_interval_derived"],
        dropna=False,
    )

    summary = {
        "format": "IRIS_SEP_PUBLISHED_SEPNET_EPISODE_AUDIT_V1",
        "upstream": {
            "repository": UPSTREAM_REPO,
            "commit": UPSTREAM_COMMIT,
            "source_manifest": manifest,
        },
        "claim_boundary": {
            "published_final_test_score_reproduced": False,
            "published_paper_called_biased": False,
            "purpose": "audit target/window construction and physical-episode multiplicity on the pinned public data tables",
        },
        "window_table": {
            "rows": int(len(frame)),
            "window_hours": 24,
            "recorded_operational_positive_windows": int((frame["recorded_operational_label"] == 1).sum()),
            "recorded_operational_negative_windows": int((frame["recorded_operational_label"] == 0).sum()),
        },
        "operational_episode_table": {
            "complete_operational_episodes": int(len(episode_table)),
            "episodes_mapped_to_at_least_one_recorded_positive_window": mapped_distinct,
            "episode_duration_hours_median": float(episode_table["duration_hours"].median()),
            "episode_duration_hours_max": float(episode_table["duration_hours"].max()),
        },
        "multiplicity": {
            "uniquely_mapped_recorded_positive_windows": int(len(uniquely_mapped)),
            "zero_episode_match_recorded_positive_windows": int((positive["occurrence_episode_match_count"] == 0).sum()),
            "multiple_episode_match_recorded_positive_windows": int((positive["occurrence_episode_match_count"] > 1).sum()),
            "episode_multiplicity_factor_uniquely_mapped": emf,
            "median_positive_windows_per_mapped_episode": float(per_episode["positive_window_count"].median()) if len(per_episode) else float("nan"),
            "max_positive_windows_per_mapped_episode": int(per_episode["positive_window_count"].max()) if len(per_episode) else 0,
            "mapped_episodes_with_more_than_one_positive_window": int((per_episode["positive_window_count"] > 1).sum()) if len(per_episode) else 0,
            "mapped_episodes_with_at_least_three_positive_windows": int((per_episode["positive_window_count"] >= 3).sum()) if len(per_episode) else 0,
        },
        "state_decomposition": {
            "all_window_attrition": attrition_counts(frame["eligibility_code"].tolist()),
            "recorded_positive_persistence_windows": int(((frame["recorded_operational_label"] == 1) & (frame["eligibility_code"] == "ALREADY_ACTIVE_PERSISTENCE")).sum()),
            "recorded_positive_new_onset_windows": int(((frame["recorded_operational_label"] == 1) & (frame["eligibility_code"] == "ELIGIBLE_ONSET_POSITIVE")).sum()),
            "recorded_positive_onset_negative_by_threshold_intervals": int(((frame["recorded_operational_label"] == 1) & (frame["eligibility_code"] == "ELIGIBLE_ONSET_NEGATIVE")).sum()),
            "recorded_positive_persistence_fraction": float((((frame["recorded_operational_label"] == 1) & (frame["eligibility_code"] == "ALREADY_ACTIVE_PERSISTENCE")).sum()) / len(positive)) if len(positive) else float("nan"),
        },
        "target_reconciliation": {
            "recorded_vs_threshold_interval_mismatch_count": int(len(mismatch)),
            "mismatch_fraction": float(len(mismatch) / len(frame)) if len(frame) else float("nan"),
            "crosstab": {
                str(r): {str(c): int(label_crosstab.loc[r, c]) for c in label_crosstab.columns}
                for r in label_crosstab.index
            },
        },
        "weights": {
            "recorded_positive_standard_total_weight": float((frame["recorded_operational_label"] == 1).sum()),
            "episode_normalized_occurrence_positive_total_weight": float(frame.loc[frame["recorded_operational_label"] == 1, "episode_normalized_occurrence_weight"].sum()),
            "expected_normalized_positive_weight_equals_uniquely_mapped_distinct_episodes": mapped_distinct,
        },
        "protected_post_2025_outcomes_accessed": False,
        "model_scores_computed": False,
    }

    frame.to_csv(out / "published_window_audit.csv", index=False)
    episode_table.to_csv(out / "published_episode_table.csv", index=False)
    per_episode.to_csv(out / "published_multiplicity_by_episode.csv", index=False)
    label_crosstab.to_csv(out / "published_target_reconciliation_crosstab.csv")
    dump_json(out / "summary.json", summary)
    dump_json(out / "source_manifest.json", manifest)

    evidence = {
        "summary_sha256": sha256_bytes((out / "summary.json").read_bytes()),
        "window_audit_sha256": sha256_bytes((out / "published_window_audit.csv").read_bytes()),
        "episode_table_sha256": sha256_bytes((out / "published_episode_table.csv").read_bytes()),
        "multiplicity_table_sha256": sha256_bytes((out / "published_multiplicity_by_episode.csv").read_bytes()),
        "source_manifest_sha256": sha256_bytes((out / "source_manifest.json").read_bytes()),
    }
    dump_json(out / "evidence_hashes.json", evidence)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = audit(args.output)
    print(json.dumps({
        "positive_windows": summary["window_table"]["recorded_operational_positive_windows"],
        "mapped_episodes": summary["operational_episode_table"]["episodes_mapped_to_at_least_one_recorded_positive_window"],
        "emf": summary["multiplicity"]["episode_multiplicity_factor_uniquely_mapped"],
        "persistence_fraction": summary["state_decomposition"]["recorded_positive_persistence_fraction"],
        "label_mismatches": summary["target_reconciliation"]["recorded_vs_threshold_interval_mismatch_count"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
