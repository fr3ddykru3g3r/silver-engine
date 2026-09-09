"""Training-only structural diagnosis for the pinned IRIS-SEP V3 development table.

This script intentionally reads the table only to bind its schema and then
filters to role=train before calculating any label or feature diagnostic.
It never loads the outer-role label/feature values for analysis and does not
fit, tune, score, or access a locked test file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data_processed" / "sepnet_v1_development_v3.csv"
MANIFEST = ROOT / "receipts" / "sepnet_v1_development_v3_manifest.json"
EXPECTED_CSV = "ab2bef52a80ebce5c27d2312f031b410843b3fa8e6b351d07a02f3e0ded010ef"
EXPECTED_MANIFEST = "18c10d4fc76a2ce5e03b9a271951003f274435aa00180fcb90e4f2947eedaebb"
EXPECTED_FEATURE = "7bca82f223f1be0adbd8afc6e30aed238ed52b3bb2339a98fa9c9cbd944436b5"
TARGET = "future_Operational_SEP_label"
META = {"issue_id", "role", "unit_id", "window_begin", "window_end", TARGET}
ROLES = {"train", "validation_monitor", "validation_calibration", "validation_threshold"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pct(n: int, d: int) -> float:
    return float(n / d) if d else 0.0


def run(csv: Path = CSV, manifest_path: Path = MANIFEST) -> dict:
    csv_hash, manifest_hash = sha256(csv), sha256(manifest_path)
    if csv_hash != EXPECTED_CSV or manifest_hash != EXPECTED_MANIFEST:
        raise RuntimeError("pinned V3 CSV or manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("locked_test_rows_present") is not False:
        raise RuntimeError("manifest does not bind a non-locked development cohort")

    frame = pd.read_csv(csv, low_memory=False)
    if not META.issubset(frame.columns) or set(frame["role"].dropna().unique()) != ROLES:
        raise RuntimeError("V3 schema/role contract failure")
    features = [c for c in frame.columns if c not in META]
    feature_hash = hashlib.sha256(json.dumps(features, separators=(",", ":")).encode()).hexdigest()
    if feature_hash != EXPECTED_FEATURE or any(c.lower().startswith("future_") for c in features):
        raise RuntimeError("feature allowlist failure")

    # Bind role membership, then discard all outer-role rows before analysis.
    train = frame.loc[frame["role"].eq("train")].copy()
    train["window_end"] = pd.to_datetime(train["window_end"], utc=True, errors="raise")
    train["quarter"] = train["window_end"].dt.to_period("Q").astype(str)
    train["label"] = pd.to_numeric(train[TARGET], errors="raise").astype(int)
    if not train["label"].isin([0, 1]).all():
        raise RuntimeError("non-binary training label")
    unit_counts = train.groupby("unit_id").agg(label=("label", "first"), rows=("label", "size"), mixed=("label", "nunique"))
    if (unit_counts["mixed"] != 1).any():
        raise RuntimeError("mixed-label training unit")

    quarters = []
    for quarter, group in train.groupby("quarter", sort=True):
        units = unit_counts.loc[group["unit_id"].unique()]
        quarters.append({
            "quarter": quarter, "rows": int(len(group)), "units": int(len(units)),
            "positive_rows": int(group["label"].sum()), "positive_prevalence": float(group["label"].mean()),
            "event_units": int((units["label"] == 1).sum()), "quiet_units": int((units["label"] == 0).sum()),
        })

    numeric = train[features].apply(pd.to_numeric, errors="coerce")
    missing_counts = numeric.isna().sum()
    missing_rate = (missing_counts / len(train)).sort_values(ascending=False)
    # Exact duplicate predictors are checked after numeric coercion; NaNs compare equal.
    duplicate_groups: dict[str, list[str]] = {}
    seen: dict[bytes, str] = {}
    for col in features:
        key = pd.util.hash_pandas_object(numeric[col], index=False).values.tobytes()
        if key in seen:
            duplicate_groups.setdefault(seen[key], []).append(col)
        else:
            seen[key] = col
    nearconstant = []
    for col in features:
        observed = numeric[col].dropna()
        if observed.empty or observed.nunique(dropna=True) <= 1:
            nearconstant.append({"feature": col, "observed_nunique": int(observed.nunique(dropna=True)), "missing_rate": float(missing_rate[col])})

    modality_columns = {
        "magnetic_sharp": [c for c in features if c == "sharp_label" or c.startswith("sharp_")],
        "eruption_flare": [c for c in features if c.startswith("flare_")],
        "cme": [c for c in features if c.startswith("CME_")],
    }
    modality = {}
    for name, cols in modality_columns.items():
        observed = numeric[cols].notna().any(axis=1)
        by_q = train.assign(_observed=observed).groupby("quarter")
        modality[name] = {
            "feature_count": len(cols), "row_available": int(observed.sum()),
            "row_missing": int((~observed).sum()), "row_missing_rate": float((~observed).mean()),
            "quarter_missing_rate": {q: float((~g["_observed"]).mean()) for q, g in by_q},
            "feature_missing_rate": {c: float(missing_rate[c]) for c in cols},
        }

    # Era comparison is descriptive; the midpoint is fixed from the training chronology.
    dates = train["window_end"]
    cutoff = dates.min() + (dates.max() - dates.min()) / 2
    era = {"early": dates <= cutoff, "late": dates > cutoff}
    era_missingness = {}
    for label, mask in era.items():
        era_missingness[label] = {
            "window_end_min": dates[mask].min().isoformat(), "window_end_max": dates[mask].max().isoformat(),
            "rows": int(mask.sum()),
            "modality_missing_rate": {name: float((~numeric.loc[mask, cols].notna().any(axis=1)).mean()) for name, cols in modality_columns.items()},
            "feature_missingness_top10": [
                {"feature": c, "missing_rate": float(numeric.loc[mask, c].isna().mean())}
                for c in numeric.loc[mask].isna().mean().sort_values(ascending=False).head(10).index
            ],
        }

    result = {
        "status": "PASS_TRAINING_ONLY_DIAGNOSIS",
        "source_csv_sha256": csv_hash, "source_manifest_sha256": manifest_hash,
        "feature_manifest_sha256": feature_hash, "target_semantics": manifest["target_semantics"],
        "locked_test_accessed": False, "outer_validation_values_inspected": False,
        "train_rows": int(len(train)), "train_units": int(len(unit_counts)),
        "train_window_end": {"min": dates.min().isoformat(), "max": dates.max().isoformat()},
        "label_summary": {"positive_rows": int(train["label"].sum()), "negative_rows": int((train["label"] == 0).sum()), "prevalence": float(train["label"].mean()), "event_units": int((unit_counts["label"] == 1).sum()), "quiet_units": int((unit_counts["label"] == 0).sum())},
        "chronological_quarters": quarters, "modality_missingness": modality, "era_missingness": era_missingness,
        "feature_missingness_top20": [{"feature": c, "missing_count": int(missing_counts[c]), "missing_rate": float(missing_rate[c])} for c in missing_rate.head(20).index],
        "label_like_feature_counts": {c: {str(k): int(v) for k, v in train[c].value_counts(dropna=False).items()} for c in ["sharp_label", "flare_label", "CME_label"]},
        "exact_duplicate_predictor_groups": duplicate_groups,
        "nearconstant_features": nearconstant,
        "limitations": [
            "The legacy future_Operational_SEP_label is a publisher window label, not the audited NEW crossing target.",
            "This describes training coverage and structure only; it contains no model fit, tuning, score, or causal attribution.",
            "The table has magnetic SHARP, flare, and CME-derived predictors but no particle-context branch; era-dependent missingness may limit a single stationary model.",
        ],
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path(__file__).with_name("training_diagnosis.json"))
    args = ap.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "train_rows": result["train_rows"], "train_units": result["train_units"]}, sort_keys=True))


if __name__ == "__main__":
    main()
