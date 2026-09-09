"""Compound-fault and recovery benchmark for admission V2."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from iris_report.iris_sep.src.iris_sep.pilot_admission_v2 import AdmissionPolicyV2, replay_forecast_v2
from iris_report.iris_sep.tests.test_pilot_replay import fixture


FAULTS = ("stale", "future_publication", "critical_missing", "optional_missing",
          "schema", "evidence", "model_binding", "uncertainty", "unsupported_era",
          "source_revision", "magnitude_shift", "nonfinite_output")


def policy() -> AdmissionPolicyV2:
    return AdmissionPolicyV2(datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2027, 1, 1, tzinfo=timezone.utc),
        {name: ("fixture-v1",) for name in ("magnetic", "eruption", "particle_context")}, 20.0)


def clean() -> tuple[dict, list[float], list[float]]:
    request = fixture()
    for record in request["data_freshness"].values():
        record["source_revision"] = "fixture-v1"
    return request, [3.0, -2.0], [0.3]


def inject(request: dict, features: list[float], outputs: list[float], fault: str) -> None:
    if fault == "stale": request["data_freshness"]["magnetic"]["observed_at_utc"] = "2025-12-31T00:00:00Z"
    elif fault == "future_publication": request["data_freshness"]["eruption"]["published_at_utc"] = "2026-01-01T12:01:00Z"
    elif fault == "critical_missing": request["missing_modalities"] = sorted(set(request["missing_modalities"]) | {"magnetic"})
    elif fault == "optional_missing": request["missing_modalities"] = sorted(set(request["missing_modalities"]) | {"particle_context"})
    elif fault == "schema": request["input_schema_sha256"] = "c" * 64
    elif fault == "evidence": request["evidence_bytes"] += b"mutation"
    elif fault == "model_binding": request["model_version"] = "unbound-model"
    elif fault == "uncertainty": request["uncertainty"].pop("input_quality", None)
    elif fault == "unsupported_era": request["issued_at"] = datetime(2028, 1, 1, tzinfo=timezone.utc)
    elif fault == "source_revision": request["data_freshness"]["magnetic"]["source_revision"] = "unknown"
    elif fault == "magnitude_shift": features[0] = 1000.0
    elif fault == "nonfinite_output": outputs[0] = float("nan")


def run(output: Path, trials: int, seed: int) -> dict:
    if output.exists(): raise ValueError("immutable output exists")
    rng = np.random.default_rng(seed); unsafe = wrong = 0; counts = {}; examples = []
    for trial in range(trials):
        request, features, outputs = clean()
        count = int(rng.integers(0, 4))
        selected = sorted(rng.choice(FAULTS, size=count, replace=False).tolist()) if count else []
        for fault in selected: inject(request, features, outputs, fault)
        only_optional = selected == ["optional_missing"]
        expected = "VALID" if not selected else ("DEGRADED" if only_optional else "ABSTAIN")
        result = replay_forecast_v2(admission_policy=policy(), transformed_features=features,
                                    model_outputs=outputs, **request)
        wrong += int(result["forecast_status"] != expected)
        unsafe += int(expected == "ABSTAIN" and result["forecast_status"] != "ABSTAIN")
        key = "+".join(selected) if selected else "none"
        counts[key] = counts.get(key, 0) + 1
        if trial < 100: examples.append({"faults": selected, "expected": expected,
            "observed": result["forecast_status"], "reasons": result["abstention_reasons"]})
    recovery_failures = 0
    for fault in FAULTS:
        broken_request, broken_features, broken_outputs = clean(); inject(broken_request, broken_features, broken_outputs, fault)
        broken = replay_forecast_v2(admission_policy=policy(), transformed_features=broken_features,
                                    model_outputs=broken_outputs, **broken_request)
        recovered_request, recovered_features, recovered_outputs = clean()
        recovered = replay_forecast_v2(admission_policy=policy(), transformed_features=recovered_features,
                                       model_outputs=recovered_outputs, **recovered_request)
        if fault != "optional_missing" and broken["forecast_status"] != "ABSTAIN": recovery_failures += 1
        if recovered["forecast_status"] != "VALID": recovery_failures += 1
    result = {"scope":"SYNTHETIC_COMPOUND_SOFTWARE_VALIDITY_NOT_SEP_SKILL",
        "trials":trials,"seed":seed,"unique_fault_combinations":len(counts),
        "status_errors":wrong,"unsafe_valid_outputs":unsafe,"recovery_sequences":len(FAULTS),
        "recovery_failures":recovery_failures,"combination_counts":counts,"examples":examples,
        "locked_test_accessed":False,"scientific_superiority_established":False,
        "claim_boundary":"Synthetic contract and recovery behavior only; no named competitor comparison."}
    output.mkdir(parents=True);data=json.dumps(result,indent=2,allow_nan=False).encode()+b"\n"
    (output/"result.json").write_bytes(data)
    (output/"receipt.json").write_text(json.dumps({"result_sha256":hashlib.sha256(data).hexdigest(),
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"locked_test_accessed":False},indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trials",type=int,default=10_000);parser.add_argument("--seed",type=int,default=20260905)
    args=parser.parse_args();print(json.dumps(run(args.output,args.trials,args.seed),indent=2))
