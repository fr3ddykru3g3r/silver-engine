from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROLLING_SHA = "4691cedd3209a2823b9e3c5e3dfe5676bde42befc1af14e33f35a220b6dfa0fb"
EVENT_SHA = "0ec9f0d6e088821091fcd369481bbbc9a2281a92fc8a582df40aefa62cae59b0"
START_COL = ">10.0 MeV 10.0 pfu SEP Start Time"
END_COL = ">10.0 MeV 10.0 pfu SEP End Time"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def classify_all(rolling: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    starts = utc(events[START_COL])
    ends = utc(events[END_COL])
    ok = starts.notna() & ends.notna() & (ends > starts)
    ep = pd.DataFrame({"start": starts[ok].to_numpy(), "end": ends[ok].to_numpy()}).reset_index(drop=True)
    ep["episode_id"] = [f"V{i:05d}" for i in range(1, len(ep) + 1)]

    rows = []
    for idx, r in rolling.iterrows():
        issue = pd.Timestamp(r["window_end"])
        horizon = issue + pd.Timedelta(hours=24)
        overlap = ep[(ep.start <= horizon) & (ep.end >= issue)]
        active = ep[(ep.start <= issue) & (ep.end > issue)]
        future = ep[(ep.start > issue) & (ep.start <= horizon)]
        ids = overlap.episode_id.tolist()
        if len(active) > 1 or len(future) > 1:
            code = "DUPLICATE_OR_AMBIGUOUS"
        elif len(active) == 1:
            code = "ALREADY_ACTIVE_PERSISTENCE"
        elif len(future) == 1:
            code = "ELIGIBLE_ONSET_POSITIVE"
        else:
            code = "ELIGIBLE_ONSET_NEGATIVE"
        rows.append({
            "upstream_row_index": int(idx),
            "issue_timestamp": issue,
            "recorded_operational_label": int(r["Future_OSEP_label"]),
            "interval_occurrence_label": int(bool(ids)),
            "eligibility_code": code,
            "occurrence_episode_match_count": len(ids),
            "verifier_episode_id_unique": ids[0] if len(ids) == 1 else None,
        })
    return pd.DataFrame(rows)


def mult(frame: pd.DataFrame) -> tuple[int, int, float | None, Counter]:
    pos = frame[frame.interval_occurrence_label == 1]
    unique = pos[pos.occurrence_episode_match_count == 1]
    counts = Counter(unique.verifier_episode_id_unique.dropna().astype(str))
    nwin = len(unique)
    nep = len(counts)
    return nwin, nep, (nwin / nep if nep else None), counts


def verify(root: Path) -> dict:
    problems: list[str] = []
    rolling_path = root / "upstream_rolling_combinded_seq_24hours.csv"
    event_path = root / "upstream_GOES_operational_event_catalog.csv"
    if sha(rolling_path) != ROLLING_SHA:
        problems.append("rolling source sha mismatch")
    if sha(event_path) != EVENT_SHA:
        problems.append("event source sha mismatch")

    manifest = json.loads((root / "evidence_hashes.json").read_text())
    hash_bad = []
    for name, expected in manifest.items():
        p = root / name
        if not p.is_file() or sha(p) != expected:
            hash_bad.append(name)
    if hash_bad:
        problems.append("evidence hash mismatch: " + ",".join(hash_bad))

    rolling = pd.read_csv(rolling_path, low_memory=False)
    events = pd.read_csv(event_path, low_memory=False)
    rolling["window_begin"] = utc(rolling["window_begin"])
    rolling["window_end"] = utc(rolling["window_end"])
    if len(rolling) != 14464:
        problems.append(f"row count {len(rolling)} != 14464")
    if int(pd.to_numeric(rolling["Future_OSEP_label"], errors="raise").sum()) != 650:
        problems.append("stored positive count != 650")
    if not np.allclose(((rolling.window_end - rolling.window_begin).dt.total_seconds()/3600).to_numpy(float), 24.0):
        problems.append("non-24-hour predictor windows")

    recomputed = classify_all(rolling, events)
    persisted = pd.read_csv(root / "sep_prism_window_audit.csv")
    persisted["issue_timestamp"] = utc(persisted["issue_timestamp"])
    cols = ["upstream_row_index", "issue_timestamp", "recorded_operational_label", "interval_occurrence_label", "eligibility_code", "occurrence_episode_match_count"]
    left = persisted[cols].sort_values("upstream_row_index").reset_index(drop=True)
    right = recomputed[cols].sort_values("upstream_row_index").reset_index(drop=True)
    if not left.equals(right):
        problems.append("row-level audit does not independently reproduce")

    summary = json.loads((root / "summary.json").read_text())
    nwin, nep, emf, counts = mult(recomputed)
    state = Counter(recomputed.loc[recomputed.interval_occurrence_label == 1, "eligibility_code"].astype(str))
    mismatch = int((recomputed.recorded_operational_label != recomputed.interval_occurrence_label).sum())
    primary = summary["primary_reconstructed_multiplicity"]
    physical = summary["physical_state_decomposition"]
    checks = {
        "rows": (len(recomputed), summary["window_table"]["rows"]),
        "reconstructed_positive_windows": (int(recomputed.interval_occurrence_label.sum()), physical["reconstructed_positive_windows"]),
        "unique_positive_windows": (nwin, primary["uniquely_mapped_positive_windows"]),
        "represented_episodes": (nep, primary["represented_physical_episodes"]),
        "persistence": (int(state["ALREADY_ACTIVE_PERSISTENCE"]), physical["already_active_persistence_windows"]),
        "onset": (int(state["ELIGIBLE_ONSET_POSITIVE"]), physical["new_onset_windows"]),
        "mismatch": (mismatch, summary["target_reconciliation"]["mismatch_count"]),
    }
    for name, pair in checks.items():
        if pair[0] != pair[1]:
            problems.append(f"summary mismatch {name}: {pair}")
    reported_emf = primary["episode_multiplicity_factor"]
    if not np.isclose(float(emf), float(reported_emf), atol=1e-12, rtol=0):
        problems.append(f"emf mismatch: {emf} != {reported_emf}")

    # Independently verify normalized physical positive mass = number of represented episodes.
    weight_mass = 0.0
    for episode_id, count in counts.items():
        weight_mass += count * (1.0 / count)
    if not np.isclose(weight_mass, nep, atol=1e-12, rtol=0):
        problems.append("independent normalized positive mass does not equal represented episode count")
    if not np.isclose(float(summary["weights"]["episode_normalized_reconstructed_positive_mass"]), nep, atol=1e-9, rtol=0):
        problems.append("reported normalized positive mass does not equal represented episodes")

    strong = bool(emf is not None and emf >= 1.5 and state["ALREADY_ACTIVE_PERSISTENCE"] > state["ELIGIBLE_ONSET_POSITIVE"])
    expected_status = "STRONG_CONFIRMATION" if strong else ("PARTIAL_CONFIRMATION" if (emf is not None and emf > 1.0) or state["ALREADY_ACTIVE_PERSISTENCE"] > 0 else "NULL_OR_REFUTATION")
    if summary["status"] != expected_status:
        problems.append(f"disposition mismatch: {summary['status']} != {expected_status}")

    return {
        "format": "IRIS_SEP_PRISM_EXTERNAL_CONFIRMATION_INDEPENDENT_VERIFICATION_V1",
        "passed": not problems,
        "problems": problems,
        "recomputed": {
            "rows": len(recomputed),
            "stored_positive_windows": int(recomputed.recorded_operational_label.sum()),
            "reconstructed_positive_windows": int(recomputed.interval_occurrence_label.sum()),
            "uniquely_mapped_reconstructed_positive_windows": nwin,
            "represented_physical_episodes": nep,
            "episode_multiplicity_factor": emf,
            "persistence_windows": int(state["ALREADY_ACTIVE_PERSISTENCE"]),
            "new_onset_windows": int(state["ELIGIBLE_ONSET_POSITIVE"]),
            "target_mismatch_count": mismatch,
            "expected_disposition": expected_status,
        },
        "source_sha256": {"rolling": sha(rolling_path), "events": sha(event_path)},
        "evidence_files_verified": len(manifest),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("artifact_dir", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = verify(a.artifact_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if a.output:
        a.output.write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
