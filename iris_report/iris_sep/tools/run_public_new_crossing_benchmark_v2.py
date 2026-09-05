"""Pre-result hardening for the public IRIS-SEP NEW-crossing diagnostic.

This wrapper corrects two methodological issues found before any scientific
score from the public benchmark was observed:

1. Quiet units are contiguous negative blocks. They break at every positive
   event, at long cadence gaps, and after seven days. A calendar block can
   therefore never straddle an event episode.
2. CLEAR-derived current-window OSEP/GSEP catalogue labels are excluded from
   learned model features. They may appear only in the explicitly labelled
   causal-persistence comparator. Learned models use measured proton flux.

All model hyperparameters, seeds, calibration, thresholding, score-role use and
10,000-unit bootstrap settings remain exactly those in v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1


CATALOGUE_PREDICTOR_LABELS = frozenset({"OSEP_label", "GSEP_label"})
MAX_CONTIGUOUS_GAP = pd.Timedelta(hours=36)


def build_units(issue_times, targets, event_ids):
    """Build contiguous episode/quiet units, then chronological purged roles."""
    issue_times = pd.Series(issue_times).reset_index(drop=True)
    targets = np.asarray(targets, dtype=np.int8)
    event_ids = np.asarray(event_ids, dtype=str)
    if not (len(issue_times) == len(targets) == len(event_ids)) or len(targets) == 0:
        raise ValueError("unit inputs must be aligned and non-empty")

    units: list[str] = []
    current_quiet: str | None = None
    quiet_start = None
    previous_time = None
    quiet_number = 0

    for t, y, event_id in zip(issue_times, targets, event_ids):
        if int(y) == 1:
            if not event_id:
                raise ValueError("positive NEW-crossing row lacks event identity")
            uid = f"event-{event_id}"
            current_quiet = None
            quiet_start = None
        else:
            starts_new_quiet = (
                current_quiet is None
                or previous_time is None
                or t - previous_time > MAX_CONTIGUOUS_GAP
                or (quiet_start is not None and t - quiet_start >= pd.Timedelta(days=v1.QUIET_BLOCK_DAYS))
            )
            if starts_new_quiet:
                token = hashlib.sha256(t.isoformat().encode("ascii")).hexdigest()[:20]
                current_quiet = f"quiet-{quiet_number:05d}-{token}"
                quiet_number += 1
                quiet_start = t
            uid = current_quiet
        units.append(uid)
        previous_time = t

    table = pd.DataFrame(
        {
            "row": np.arange(len(units)),
            "unit_id": units,
            "time": issue_times,
            "label": targets,
        }
    )
    if (table.groupby("unit_id")["label"].nunique() > 1).any():
        raise ValueError("unit mixes positive and negative rows")
    summaries = (
        table.groupby("unit_id", sort=False)
        .agg(start=("time", "min"), end=("time", "max"), label=("label", "max"))
        .sort_values(["start", "end"])
    )
    if summaries.index.duplicated().any():
        raise ValueError("unit identities are not unique")

    positive_units = int(summaries["label"].sum())
    if positive_units < 20:
        raise ValueError(f"too few positive event units: {positive_units}")
    cutoffs = (
        int(positive_units * 0.70),
        int(positive_units * 0.80),
        int(positive_units * 0.90),
    )

    mapping: dict[str, str] = {}
    positives_seen = 0
    for uid, row in summaries.iterrows():
        if positives_seen < cutoffs[0]:
            role = "fit"
        elif positives_seen < cutoffs[1]:
            role = "calibration"
        elif positives_seen < cutoffs[2]:
            role = "threshold"
        else:
            role = "score"
        mapping[str(uid)] = role
        positives_seen += int(row.label)

    # Purge complete units from the beginning of every right-hand block until
    # its first retained timestamp is strictly more than 24 h after the prior
    # block's final retained timestamp.
    purged: set[str] = set()
    for left, right in zip(v1.ROLE_ORDER, v1.ROLE_ORDER[1:]):
        left_units = [u for u, role in mapping.items() if role == left and u not in purged]
        if not left_units:
            raise ValueError(f"empty role before purge: {left}")
        left_end = summaries.loc[left_units, "end"].max()
        right_units = [u for u, role in mapping.items() if role == right]
        for uid in right_units:
            if summaries.loc[uid, "start"] <= left_end + pd.Timedelta(hours=v1.PURGE_HOURS):
                purged.add(uid)

    roles = np.array(
        [mapping[u] if u not in purged else "purged" for u in units],
        dtype="U16",
    )
    for role in v1.ROLE_ORDER:
        labels = targets[roles == role]
        if len(labels) == 0 or len(np.unique(labels)) != 2:
            raise ValueError(f"role {role} lacks both classes after purge")
    for left, right in zip(v1.ROLE_ORDER, v1.ROLE_ORDER[1:]):
        gap = issue_times[roles == right].min() - issue_times[roles == left].max()
        if not gap > pd.Timedelta(hours=v1.PURGE_HOURS):
            raise ValueError(f"strict purge failed: {left}->{right}: {gap}")

    # Final chronology invariant: no retained role may reappear after a later role.
    role_rank = {role: i for i, role in enumerate(v1.ROLE_ORDER)}
    retained = [role_rank[r] for r in roles if r != "purged"]
    if any(b < a for a, b in zip(retained, retained[1:])):
        raise ValueError("retained roles interleave chronologically")
    return np.asarray(units, dtype="U64"), roles, sorted(purged), positive_units


def feature_sets(frame):
    """Predeclared context families without retrospective catalogue labels."""
    excluded = {"window_begin", "window_end", *CATALOGUE_PREDICTOR_LABELS}
    feature_names: list[str] = []
    dropped_non_numeric: list[str] = []
    for column in frame.columns:
        low = column.lower()
        if column in excluded or low.startswith("future_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            feature_names.append(column)
        else:
            dropped_non_numeric.append(column)
    if not feature_names:
        raise ValueError("no numeric causal predictors")

    proton = [column for column in feature_names if "protonflux" in column.lower()]
    xrs = [column for column in feature_names if "xrs" in column.lower()]
    context = set(proton + xrs)
    base = [column for column in feature_names if column not in context]
    if not proton or not xrs or not base:
        raise ValueError("expected base, measured-proton and XRS feature families")
    return {
        "BASE_SOLAR": base,
        "BASE_PLUS_PROTON": base + proton,
        "BASE_PLUS_XRS": base + xrs,
        "FULL_CONTEXT": base + proton + xrs,
    }, dropped_non_numeric


def run(features: Path, events: Path, output: Path):
    # Patch only the two pre-result methods above; every numerical/model method
    # remains frozen in v1.
    v1.build_units = build_units
    v1.feature_sets = feature_sets
    result = v1.run(features, events, output)
    receipt = {
        "status": "PRE_RESULT_METHOD_HARDENING_APPLIED",
        "parent_runner": "run_public_new_crossing_benchmark.py",
        "changes": [
            "CONTIGUOUS_NEGATIVE_QUIET_UNITS_BREAK_AT_EVENTS_GAPS_AND_7_DAYS",
            "CLEAR_OSEP_GSEP_LABELS_EXCLUDED_FROM_LEARNED_MODEL_FEATURES",
        ],
        "unchanged": [
            "TARGET_SEMANTICS",
            "MODEL_HYPERPARAMETERS",
            "XGBOOST_SEEDS",
            "CALIBRATION_ROLE",
            "THRESHOLD_ROLE",
            "SCORE_ROLE",
            "BOOTSTRAP_REPLICATES",
        ],
        "catalogue_labels_excluded_from_learned_models": sorted(CATALOGUE_PREDICTOR_LABELS),
        "catalogue_label_allowed_only_in_named_comparator": "CAUSAL_PERSISTENCE",
        "scientific_score_seen_before_change": False,
        "locked_test_accessed": False,
    }
    Path(output, "v2_method_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.events, args.output)


if __name__ == "__main__":
    main()
