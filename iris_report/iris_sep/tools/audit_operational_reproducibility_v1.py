#!/usr/bin/env python3
"""Audit evidence for operational reproduction of the exact frozen SEP interface.

This audit is outcome-blind. It classifies the exact predictor columns from a pinned
schema against a source-family evidence manifest. `VERIFIED` means the historical
feature construction itself has issue-time evidence; `UNVERIFIED_*` means evidence
is incomplete, not that the physical measurement did not exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

VALID_STATUSES = {"VERIFIED","UNVERIFIED_LATENCY","SCHEMA_MISMATCH","RETROSPECTIVE_ONLY","NO_EQUIVALENT"}
RAW_TABLE_EXCLUDED_EXACT = {"window_begin", "window_end", "OSEP_label", "GSEP_label"}
RAW_TABLE_EXCLUDED_PREFIXES = ("Future_",)


def ordered_hash(columns: list[str]) -> str:
    payload = json.dumps(columns, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_columns(path: Path) -> tuple[list[str], dict]:
    """Return exact predictors. Present-time *_label fields are deliberately retained."""
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            cols, meta = obj, {}
        elif isinstance(obj, dict) and isinstance(obj.get("columns"), list):
            cols, meta = obj["columns"], obj
        else:
            raise ValueError("JSON must be a list or an object containing 'columns'.")
        if not all(isinstance(c, str) for c in cols):
            raise ValueError("All schema columns must be strings")
        expected_hash = meta.get("ordered_columns_sha256")
        if expected_hash and ordered_hash(cols) != expected_hash:
            raise ValueError("ordered feature hash mismatch")
        return cols, meta
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV is empty") from exc
    cols = [c for c in header if c not in RAW_TABLE_EXCLUDED_EXACT and not any(c.startswith(p) for p in RAW_TABLE_EXCLUDED_PREFIXES)]
    return cols, {"schema_type": "upstream_rolling_table_header"}


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
    return sorted(families, key=lambda f: max(len(p) for p in f["prefixes"]), reverse=True)


def classify(name: str, families: list[dict]) -> dict | None:
    candidates = []
    for fam in families:
        matched = [p for p in fam["prefixes"] if name.startswith(p)]
        if matched:
            candidates.append((max(map(len, matched)), fam))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def audit(columns: list[str], families: list[dict]) -> dict:
    rows, unmapped = [], []
    for col in columns:
        fam = classify(col, families)
        if fam is None:
            unmapped.append(col)
            continue
        rows.append({"feature": col, "family": fam["family"], "status": fam["status"], "reason": fam.get("reason", "")})
    if unmapped:
        raise ValueError("Unmapped predictor columns (fail-closed): " + ", ".join(unmapped[:25]))
    status_counts = Counter(r["status"] for r in rows)
    family_counts = Counter(r["family"] for r in rows)
    total = len(rows)
    verified = status_counts.get("VERIFIED", 0)
    directly_non_equivalent = status_counts.get("SCHEMA_MISMATCH", 0) + status_counts.get("RETROSPECTIVE_ONLY", 0) + status_counts.get("NO_EQUIVALENT", 0)
    unresolved = total - verified - directly_non_equivalent
    return {
        "predictor_count": total,
        "ordered_columns_sha256": ordered_hash(columns),
        "verified_predictor_count": verified,
        "operationally_reproducible_feature_fraction": verified / total if total else None,
        "directly_non_equivalent_or_retrospective_count": directly_non_equivalent,
        "directly_non_equivalent_or_retrospective_fraction": directly_non_equivalent / total if total else None,
        "unresolved_predictor_count": unresolved,
        "unresolved_predictor_fraction": unresolved / total if total else None,
        "exact_interface_reproducibility_upper_bound_if_all_unresolved_later_verify": (verified + unresolved) / total if total else None,
        "status_counts": dict(sorted(status_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "features": rows,
        "interpretation_guardrail": "UNVERIFIED does not mean unavailable. Fractions describe documentary/semantic reproducibility of the exact frozen interface, not sensor existence or model invalidity.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-predictors", type=int, default=259)
    ap.add_argument("--expected-ordered-hash", default=None)
    args = ap.parse_args()
    columns, schema_meta = load_columns(args.schema)
    result = audit(columns, load_manifest(args.manifest))
    result["expected_predictor_count"] = args.expected_predictors
    result["predictor_count_matches_expected"] = result["predictor_count"] == args.expected_predictors
    if not result["predictor_count_matches_expected"]:
        raise ValueError(f"Predictor count {result['predictor_count']} != frozen expected {args.expected_predictors}")
    if args.expected_ordered_hash and result["ordered_columns_sha256"] != args.expected_ordered_hash:
        raise ValueError("ordered predictor hash does not match frozen expectation")
    if schema_meta.get("predictor_count") not in (None, result["predictor_count"]):
        raise ValueError("schema metadata predictor_count mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["predictor_count","ordered_columns_sha256","verified_predictor_count","operationally_reproducible_feature_fraction","directly_non_equivalent_or_retrospective_count","unresolved_predictor_count","status_counts","family_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
