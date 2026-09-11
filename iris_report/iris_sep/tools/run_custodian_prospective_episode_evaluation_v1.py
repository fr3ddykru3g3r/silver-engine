#!/usr/bin/env python3
"""Aggregate-only custodian evaluator for IRIS SEP prospective confirmation V1.

Consumes a sealed row-level custodian table and a fully frozen execution
manifest. Emits only aggregate results and an execution receipt; it never
writes row-level labels or predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

STUDY_ID = "IRIS_SEP_PROSPECTIVE_OPERATIONAL_CONFIRMATION_V1"
PROTECTED_START = datetime(2025, 9, 10, tzinfo=timezone.utc)
REQUIRED_COLUMNS = {
    "issue_time_utc", "prediction_timestamp_utc", "model_id", "alert", "probability",
    "mapped_occurrence", "onset_eligible", "onset_label", "active_at_issue",
    "episode_id", "quiet_block_id", "ambiguous_positive",
}
ALLOWED_EVALS = (
    "MAPPED_OCCURRENCE",
    "EPISODE_NORMALIZED_OCCURRENCE",
    "NEW_ONSET_CAUSAL",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp lacks timezone: {value}")
    return dt.astimezone(timezone.utc)


def bit(value: str, name: str) -> int:
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return int(value)


def prob(value: str) -> float:
    x = float(value)
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"probability outside [0,1]: {x}")
    return x


@dataclass(frozen=True)
class Row:
    issue: datetime
    pred_time: datetime
    model: str
    alert: int
    probability: float
    mapped: int
    onset_eligible: int
    onset: int
    active: int
    episode_id: str
    quiet_block_id: str
    ambiguous: int


def load_manifest(path: Path) -> dict:
    m = json.loads(path.read_text())
    if m.get("study_id") != STUDY_ID:
        raise ValueError("wrong study_id")
    if m.get("status") != "EXECUTION_FROZEN":
        raise ValueError("manifest must have status EXECUTION_FROZEN")
    if m.get("protected_pool", {}).get("not_before_utc") != "2025-09-10T00:00:00Z":
        raise ValueError("protected start changed")
    if m.get("bootstrap", {}).get("replicates") != 10000 or m.get("bootstrap", {}).get("seed") != 20260911:
        raise ValueError("bootstrap contract changed")
    floor = m.get("information_floor", {})
    if floor.get("minimum_distinct_onset_episodes") != 50 or floor.get("minimum_quiet_blocks") != 500:
        raise ValueError("information floor changed")
    models = m.get("frozen_models", {})
    if "past_proton_active_proxy" not in models:
        raise ValueError("required past_proton_active_proxy is not frozen")
    for model_id, spec in models.items():
        digest = str(spec.get("artifact_or_rule_sha256", ""))
        schema = str(spec.get("feature_schema_sha256", ""))
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError(f"invalid artifact/rule sha256 for {model_id}")
        if len(schema) != 64 or any(c not in "0123456789abcdef" for c in schema.lower()):
            raise ValueError(f"invalid feature schema sha256 for {model_id}")
        if not spec.get("features_verified_causal", False):
            raise ValueError(f"model {model_id} has unverified feature availability")
    return m


def load_rows(path: Path, frozen_models: Mapping[str, dict]) -> List[Row]:
    rows: List[Row] = []
    seen = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"sealed input missing columns: {sorted(missing)}")
        for raw in reader:
            issue = parse_utc(raw["issue_time_utc"])
            pred_time = parse_utc(raw["prediction_timestamp_utc"])
            if issue < PROTECTED_START:
                raise ValueError("row predates protected prospective pool")
            if (issue.hour, issue.minute, issue.second) != (0, 0, 0):
                raise ValueError("issue cadence is not daily 00:00 UTC")
            if pred_time > issue:
                raise ValueError("prediction was timestamped after issue_time")
            model = raw["model_id"]
            if model not in frozen_models:
                raise ValueError(f"unfrozen model in sealed input: {model}")
            r = Row(
                issue=issue,
                pred_time=pred_time,
                model=model,
                alert=bit(raw["alert"], "alert"),
                probability=prob(raw["probability"]),
                mapped=bit(raw["mapped_occurrence"], "mapped_occurrence"),
                onset_eligible=bit(raw["onset_eligible"], "onset_eligible"),
                onset=bit(raw["onset_label"], "onset_label"),
                active=bit(raw["active_at_issue"], "active_at_issue"),
                episode_id=raw["episode_id"].strip(),
                quiet_block_id=raw["quiet_block_id"].strip(),
                ambiguous=bit(raw["ambiguous_positive"], "ambiguous_positive"),
            )
            if r.onset and (not r.onset_eligible or r.active):
                raise ValueError("onset positive is not causally eligible")
            key = (r.issue.isoformat(), r.model)
            if key in seen:
                raise ValueError(f"duplicate issue/model row: {key}")
            seen.add(key)
            rows.append(r)
    return rows


def episode_multiplicities(rows: Sequence[Row], model: str) -> Counter:
    c = Counter()
    for r in rows:
        if r.model == model and r.mapped and not r.ambiguous and r.episode_id:
            c[r.episode_id] += 1
    return c


_CURRENT_MODEL_ROWS: Sequence[Row] = ()


def row_weight_and_label(r: Row, evaluand: str, mult: Mapping[str, int]) -> Tuple[float, int] | None:
    if r.ambiguous:
        return None
    if evaluand == "MAPPED_OCCURRENCE":
        if r.mapped:
            if not r.episode_id:
                return None
            return 1.0, 1
        if not r.quiet_block_id:
            return None
        return 1.0, 0
    if evaluand == "EPISODE_NORMALIZED_OCCURRENCE":
        if r.mapped:
            if not r.episode_id or mult.get(r.episode_id, 0) <= 0:
                return None
            return 1.0 / mult[r.episode_id], 1
        if not r.quiet_block_id:
            return None
        return 1.0, 0
    if evaluand == "NEW_ONSET_CAUSAL":
        if not r.onset_eligible:
            return None
        if r.onset:
            if not r.episode_id:
                return None
            onset_count = sum(
                1 for rr in _CURRENT_MODEL_ROWS
                if rr.onset_eligible and rr.onset and not rr.ambiguous and rr.episode_id == r.episode_id
            )
            return 1.0 / max(1, onset_count), 1
        if not r.quiet_block_id:
            return None
        return 1.0, 0
    raise ValueError(evaluand)


def metrics(rows: Sequence[Row], model: str, evaluand: str, selected_episodes=None, selected_blocks=None) -> dict:
    global _CURRENT_MODEL_ROWS
    model_rows = [r for r in rows if r.model == model]
    _CURRENT_MODEL_ROWS = model_rows
    mult = episode_multiplicities(rows, model)
    ep_mult = Counter(selected_episodes or []) if selected_episodes is not None else None
    block_mult = Counter(selected_blocks or []) if selected_blocks is not None else None
    tp = fp = fn = tn = brier_num = total_w = 0.0
    for r in model_rows:
        wl = row_weight_and_label(r, evaluand, mult)
        if wl is None:
            continue
        w, y = wl
        if y == 1 and ep_mult is not None:
            n = ep_mult.get(r.episode_id, 0)
            if n == 0:
                continue
            w *= n
        if y == 0 and block_mult is not None:
            n = block_mult.get(r.quiet_block_id, 0)
            if n == 0:
                continue
            w *= n
        if r.alert and y:
            tp += w
        elif r.alert and not y:
            fp += w
        elif (not r.alert) and y:
            fn += w
        else:
            tn += w
        brier_num += w * (r.probability - y) ** 2
        total_w += w
    pod = tp / (tp + fn) if tp + fn else math.nan
    fpr = fp / (fp + tn) if fp + tn else math.nan
    tss = pod - fpr if not (math.isnan(pod) or math.isnan(fpr)) else math.nan
    far = fp / (tp + fp) if tp + fp else math.nan
    return {
        "TSS": tss, "POD": pod, "FAR": far,
        "Brier": brier_num / total_w if total_w else math.nan,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def percentile(xs: Sequence[float], p: float) -> float:
    vals = sorted(x for x in xs if not math.isnan(x))
    if not vals:
        return math.nan
    pos = (len(vals) - 1) * p
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def bootstrap_contrast(rows: Sequence[Row], model: str, left: str, right: str, seed: int, reps: int) -> dict:
    base = [r for r in rows if r.model == model and not r.ambiguous]
    episodes = sorted({r.episode_id for r in base if r.episode_id and r.onset_eligible and r.onset})
    blocks = sorted({r.quiet_block_id for r in base if r.quiet_block_id and not r.mapped and r.onset_eligible})
    if not episodes or not blocks:
        raise ValueError("no physical units available for bootstrap")
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        sampled_eps = [episodes[rng.randrange(len(episodes))] for _ in episodes]
        sampled_blocks = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        a = metrics(rows, model, left, sampled_eps, sampled_blocks)["TSS"]
        b = metrics(rows, model, right, sampled_eps, sampled_blocks)["TSS"]
        diffs.append(a - b)
    point = metrics(rows, model, left, episodes, blocks)["TSS"] - metrics(rows, model, right, episodes, blocks)["TSS"]
    return {
        "point_difference": point,
        "bootstrap_median": percentile(diffs, 0.5),
        "ci95": [percentile(diffs, 0.025), percentile(diffs, 0.975)],
        "n_episode_units": len(episodes),
        "n_quiet_blocks": len(blocks),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed-input", type=Path, required=True)
    ap.add_argument("--execution-manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--custodian-mode", action="store_true")
    args = ap.parse_args()
    if not args.custodian_mode:
        raise SystemExit("Refusing to run outside explicit --custodian-mode")

    manifest = load_manifest(args.execution_manifest)
    rows = load_rows(args.sealed_input, manifest["frozen_models"])
    model_ids = sorted(manifest["frozen_models"])
    unique_issues = sorted({r.issue for r in rows})
    if any(sum(r.model == model and r.issue == issue for r in rows) != 1 for model in model_ids for issue in unique_issues):
        raise ValueError("each frozen model must have exactly one row per issue")

    reference = model_ids[0]
    ref_rows = [r for r in rows if r.model == reference]
    onset_eps = sorted({r.episode_id for r in ref_rows if r.onset and r.onset_eligible and r.episode_id and not r.ambiguous})
    quiet_blocks = sorted({r.quiet_block_id for r in ref_rows if r.onset_eligible and not r.onset and not r.mapped and r.quiet_block_id and not r.ambiguous})
    floor = manifest["information_floor"]
    floor_met = len(onset_eps) >= floor["minimum_distinct_onset_episodes"] and len(quiet_blocks) >= floor["minimum_quiet_blocks"]

    metric_table = {model: {ev: metrics(rows, model, ev) for ev in ALLOWED_EVALS} for model in model_ids}
    contrasts = {}
    for spec in manifest["primary_contrasts"]:
        model = spec["model_slot"]
        if model not in manifest["frozen_models"]:
            continue
        if spec["id"] == "PAST_PROTON_PERSISTENCE_EFFECT":
            left, right = "NEW_ONSET_CAUSAL", "MAPPED_OCCURRENCE"
        elif spec["id"] == "JOINT_MULTIPLICITY_EFFECT":
            left, right = "EPISODE_NORMALIZED_OCCURRENCE", "MAPPED_OCCURRENCE"
        elif spec["id"] == "JOINT_PERSISTENCE_EFFECT":
            left, right = "NEW_ONSET_CAUSAL", "EPISODE_NORMALIZED_OCCURRENCE"
        else:
            raise ValueError(f"unknown contrast id {spec['id']}")
        contrasts[spec["id"]] = bootstrap_contrast(
            rows, model, left, right,
            manifest["bootstrap"]["seed"], manifest["bootstrap"]["replicates"],
        )

    required = contrasts.get("PAST_PROTON_PERSISTENCE_EFFECT")
    supported = bool(required and required["ci95"][1] < 0)
    if not floor_met:
        disposition = "INSUFFICIENT_CONFIRMATORY_INFORMATION"
    elif supported:
        disposition = "PROSPECTIVE_CONFIRMATION"
    else:
        disposition = "NO_PROSPECTIVE_CONFIRMATION"

    out = {
        "study_id": STUDY_ID,
        "disposition": disposition,
        "information_floor_met": floor_met,
        "aggregate_counts": {
            "issue_count": len(unique_issues),
            "distinct_onset_episode_units": len(onset_eps),
            "quiet_block_units": len(quiet_blocks),
            "model_count": len(model_ids),
        },
        "metrics": metric_table,
        "primary_contrasts": contrasts,
        "claim_boundary": manifest["claim_boundary"],
        "input_receipts": {
            "sealed_input_sha256": sha256_file(args.sealed_input),
            "execution_manifest_sha256": sha256_file(args.execution_manifest),
        },
        "row_level_outputs_written": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "aggregate_results.json"
    result_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    receipt = {
        "study_id": STUDY_ID,
        "aggregate_results_sha256": sha256_file(result_path),
        "sealed_input_sha256": sha256_file(args.sealed_input),
        "execution_manifest_sha256": sha256_file(args.execution_manifest),
        "row_level_outputs_written": False,
    }
    (args.output_dir / "execution_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
