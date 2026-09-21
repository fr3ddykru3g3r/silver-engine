#!/usr/bin/env python3
"""Audit whether a retrospective SEP predictor interface has operational evidence.

This tool does not inspect forecast outcomes. It classifies predictor columns using a
frozen source-family manifest and fails closed on unmapped columns.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

VALID_STATUSES = {
    "VERIFIED",
    "UNVERIFIED_LATENCY",
    "SCHEMA_MISMATCH",
    "RETROSPECTIVE_ONLY",
    "NO_EQUIVALENT",
}

NON_PREDICTOR_EXACT = {"window_begin", "window_end"}
NON_PREDICTOR_PREFIXES = ("Future_",)
NON_PREDICTOR_SUFFIXES = ("_label",)


def load_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        cols = obj["columns"] if isinstance(obj, dict) else obj
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            raise ValueError("JSON must be a list of column names or {'columns': [...]}.")
        return cols
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc


def is_predictor(name: str) -> bool:
    if name in NON_PREDICTOR_EXACT:
        return False
    if any(name.startswith(p) for p in NON_PREDICTOR_PREFIXES):
        return False
    if any(name.endswith(s) for s in NON_PREDICTOR_SUFFIXES):
        return False
    return True


def load_manifest(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    families = obj.get("families", [])
    if not families:
        raise ValueError("Manifest contains no families")
    for fam in families:
        if fam.get("status") not in VALID_STATUSES:
            raise ValueError(f"Invalid status for {fam.get('family')}: {fam.get('status')}")
        if not fam.get("prefixes"):
            raise ValueError(f"No prefixes for family {fam.get('family')}")
    # Longest prefix first prevents SHARP_ from swallowing SHARP_AR_.
    return sorted(families, key=lambda f: max(len(p) for p in f["prefixes"]), reverse=True)


def classify(name: str, families: list[dict]) -> dict | None:
    candidates = []
    for fam in families:
        matched = [p for p in fam["prefixes"] if name.startswith(p)]
        if matched:
            candidates.append((max(len(p) for p in matched), fam))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def audit(columns: list[str], families: list[dict]) -> dict:
    predictors = [c for c in columns if is_predictor(c)]
    rows = []
    unmapped = []
    for col in predictors:
        fam = classify(col, families)
        if fam is None:
            unmapped.append(col)
            continue
        rows.append({
            "feature": col,
            "family": fam["family"],
            "status": fam["status"],
            "reason": fam.get("reason", ""),
        })
    if unmapped:
        raise ValueError("Unmapped predictor columns (fail-closed): " + ", ".join(unmapped[:25]))

    status_counts = Counter(r["status"] for r in rows)
    family_counts = Counter(r["family"] for r in rows)
    total = len(rows)
    verified = status_counts.get("VERIFIED", 0)
    return {
        "predictor_count": total,
        "verified_predictor_count": verified,
        "operationally_reproducible_feature_fraction": (verified / total if total else None),
        "status_counts": dict(sorted(status_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "features": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=Path, required=True, help="CSV (header used) or JSON column list")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-predictors", type=int, default=259)
    args = ap.parse_args()

    result = audit(load_columns(args.schema), load_manifest(args.manifest))
    result["expected_predictor_count"] = args.expected_predictors
    result["predictor_count_matches_expected"] = result["predictor_count"] == args.expected_predictors
    if not result["predictor_count_matches_expected"]:
        raise ValueError(
            f"Predictor count {result['predictor_count']} != frozen expected {args.expected_predictors}; "
            "do not silently audit a different interface."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in [
        "predictor_count", "verified_predictor_count",
        "operationally_reproducible_feature_fraction", "status_counts"
    ]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
