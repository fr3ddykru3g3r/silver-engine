"""Fail-closed validation for the IRIS primary physics/downstream matrix.

The short morphology-transfer artifact downloaded from GitHub is deliberately
not accepted as the primary result.  This validator keeps the distinction
machine-checkable and validates the exact frozen science-arm mapping:

    R   -> real-only, unweighted
    Rw  -> real-only, balanced positive weighting
    D   -> duplicated real positives, unweighted
    L0  -> BASE synthetic positives, unweighted
    L2  -> Hale/Joy synthetic positives, unweighted
    L3  -> Hale/Joy + strong-PIL synthetic positives, unweighted

It is intentionally dependency-free so it can run before installing the ML
stack.  A successful validation means that an artifact is structurally
eligible for the primary analysis; it does not certify scientific quality.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PRIMARY_SCIENCE_ARMS = ("R", "Rw", "D", "L0", "L2", "L3")
IMPLEMENTATION_TO_SCIENCE = {
    "real": "R",
    "real_weighted": "Rw",
    "rw": "Rw",
    "duplicate": "D",
    "base": "L0",
    "hj": "L2",
    "hj_pil": "L3",
}
FORBIDDEN_PRIMARY_ARMS = {"pil", "pil_blur", "geometry_flip", "block_shuffle"}


def _normalise_arm(value: Any) -> str:
    raw = str(value).strip()
    return IMPLEMENTATION_TO_SCIENCE.get(raw.lower(), raw)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _arm_column(rows: Iterable[dict[str, str]]) -> str | None:
    rows = list(rows)
    if not rows:
        return None
    for key in ("science_arm", "arm", "implementation_arm", "condition"):
        if key in rows[0]:
            return key
    return None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _metric_files(root: Path) -> list[Path]:
    direct = sorted(root.glob("outputs/*/metrics.json"))
    if direct:
        return direct
    return sorted(root.rglob("metrics.json"))


def _metric_arm(path: Path, payload: dict[str, Any] | None) -> str:
    raw = (payload or {}).get("science_arm") or (payload or {}).get("arm")
    if raw is None or str(raw).strip().lower() == "synthetic":
        raw = path.parent.name
    return str(raw)


def validate_primary_artifact(root: Path) -> list[str]:
    """Return hard-failure messages for a candidate primary artifact."""

    errors: list[str] = []
    if not root.is_dir():
        return [f"primary artifact directory does not exist: {root}"]

    morphology_csvs = sorted(root.rglob("fidelity_utility_points.csv"))
    if morphology_csvs:
        rows = _read_csv(morphology_csvs[0])
        raw_arms = sorted({str(row.get("arm", "")).strip() for row in rows})
        forbidden = sorted(set(raw_arms) & FORBIDDEN_PRIMARY_ARMS)
        if forbidden:
            errors.append(
                "exploratory morphology artifact detected; forbidden primary "
                f"arms present: {', '.join(forbidden)}"
            )
        errors.append(
            "fidelity_utility_points.csv is an exploratory morphology summary, "
            "not the locked R/Rw/D/L0/L2/L3 artifact"
        )

    arm_values: list[str] = []
    primary_csvs = sorted(root.rglob("primary_metrics.csv"))
    if primary_csvs:
        rows = _read_csv(primary_csvs[0])
        key = _arm_column(rows)
        if key is None:
            errors.append("primary_metrics.csv has no arm column")
        else:
            arm_values.extend(str(row[key]).strip() for row in rows)

    metric_files = _metric_files(root)
    if not arm_values and metric_files:
        for path in metric_files:
            payload = _read_json(path)
            if payload and payload.get("arm") is not None:
                arm_values.append(_metric_arm(path, payload))
            elif path.parent.name:
                arm_values.append(path.parent.name)

    if not arm_values:
        errors.append(
            "no primary_metrics.csv or per-arm metrics.json files were found"
        )
        return errors

    normalised = [_normalise_arm(value) for value in arm_values]
    observed = set(normalised)
    expected = set(PRIMARY_SCIENCE_ARMS)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = sorted({arm for arm in normalised if normalised.count(arm) > 1})
    if missing:
        errors.append(f"missing required primary science arms: {', '.join(missing)}")
    if extra:
        errors.append(f"unexpected primary science arms: {', '.join(extra)}")
    if duplicates:
        errors.append(f"duplicate primary science-arm rows: {', '.join(duplicates)}")

    for path in metric_files:
        payload = _read_json(path)
        if not payload:
            errors.append(f"invalid metrics JSON: {path}")
            continue
        if payload.get("test_metrics_accessed_for_selection") is True:
            errors.append(f"test metrics were used for selection: {path}")
        threshold_source = payload.get("threshold_source")
        if threshold_source is not None and str(threshold_source).lower() != "validation":
            errors.append(f"threshold was not selected on validation only: {path}")

    prediction_sets: dict[str, set[str]] = {}
    for path in sorted(root.rglob("test_predictions.csv")):
        rows = _read_csv(path)
        if not rows or "sample_id" not in rows[0]:
            errors.append(f"test_predictions.csv lacks sample_id: {path}")
            continue
        science_arm = _normalise_arm(path.parent.name)
        prediction_sets[science_arm] = {str(row["sample_id"]) for row in rows}
    if prediction_sets:
        reference = prediction_sets.get("R")
        if reference is None:
            errors.append("test predictions exist but the R reference arm is missing")
        else:
            for arm, sample_ids in sorted(prediction_sets.items()):
                if sample_ids != reference:
                    errors.append(f"test sample IDs differ from R: {arm}")

    added_counts: dict[str, int] = {}
    for path in metric_files:
        payload = _read_json(path) or {}
        arm = _normalise_arm(_metric_arm(path, payload))
        if payload.get("added_positive_rows") is not None:
            try:
                added_counts[arm] = int(payload["added_positive_rows"])
            except (TypeError, ValueError):
                errors.append(f"invalid added_positive_rows for {arm}: {path}")
    required_added = {arm: added_counts.get(arm) for arm in ("D", "L0", "L2", "L3")}
    if all(value is not None for value in required_added.values()):
        values = set(required_added.values())
        if len(values) != 1 or next(iter(values)) <= 0:
            errors.append(
                "D/L0/L2/L3 do not have one identical positive-addition count"
            )

    return errors


def validate_inputs(
    evidence_dir: Path | None,
    fits_dir: Path | None,
    checkpoint: Path | None,
    expected_fits: int | None,
) -> list[str]:
    """Check that the full experiment inputs needed for continuation exist."""

    errors: list[str] = []
    if evidence_dir is not None:
        manifest = evidence_dir / "data" / "derived" / "training_manifest.csv.gz"
        if not manifest.is_file():
            errors.append(f"training evidence manifest missing: {manifest}")
    if fits_dir is not None:
        fits_count = sum(1 for path in fits_dir.rglob("*.fits") if path.is_file())
        if fits_count == 0:
            errors.append(f"no FITS files found under: {fits_dir}")
        if expected_fits is not None and fits_count != expected_fits:
            errors.append(
                f"FITS count mismatch: expected {expected_fits}, found {fits_count}"
            )
    if checkpoint is not None and not checkpoint.is_file():
        errors.append(f"BASE checkpoint missing: {checkpoint}")
    return errors


def validate(
    artifact_dir: Path | None = None,
    evidence_dir: Path | None = None,
    fits_dir: Path | None = None,
    checkpoint: Path | None = None,
    expected_fits: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    checks: list[str] = []
    if artifact_dir is not None:
        checks.append("primary_artifact")
        errors.extend(validate_primary_artifact(artifact_dir))
    if any(value is not None for value in (evidence_dir, fits_dir, checkpoint)):
        checks.append("continuation_inputs")
        errors.extend(validate_inputs(evidence_dir, fits_dir, checkpoint, expected_fits))
    if not checks:
        errors.append("no validation target was provided")
    return {"status": "PASS" if not errors else "FAIL", "checks": checks, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--fits-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-fits", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = validate(
        artifact_dir=args.artifact_dir,
        evidence_dir=args.evidence_dir,
        fits_dir=args.fits_dir,
        checkpoint=args.checkpoint,
        expected_fits=args.expected_fits,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
