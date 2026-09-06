"""Deterministic synthetic fault-injection benchmark for the pilot envelope."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from iris_report.iris_sep.src.iris_sep.pilot_replay import replay_forecast
from iris_report.iris_sep.tests.test_pilot_replay import fixture


FAULTS = ("none", "stale", "future_publication", "critical_missing",
          "optional_missing", "schema", "evidence", "model_binding",
          "uncertainty")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mutate(base: dict, fault: str) -> tuple[dict, str]:
    request = copy.deepcopy(base)
    expected = "VALID"
    if fault == "stale":
        request["data_freshness"]["magnetic"]["observed_at_utc"] = "2025-12-31T00:00:00Z"
        expected = "ABSTAIN"
    elif fault == "future_publication":
        request["data_freshness"]["eruption"]["published_at_utc"] = "2026-01-01T12:01:00Z"
        expected = "ABSTAIN"
    elif fault == "critical_missing":
        request["missing_modalities"] = ["magnetic"]
        expected = "ABSTAIN"
    elif fault == "optional_missing":
        request["missing_modalities"] = ["particle_context"]
        expected = "DEGRADED"
    elif fault == "schema":
        request["input_schema_sha256"] = "c" * 64
        expected = "ABSTAIN"
    elif fault == "evidence":
        request["evidence_bytes"] += b"mutation"
        expected = "ABSTAIN"
    elif fault == "model_binding":
        request["model_version"] = "unbound-model"
        expected = "ABSTAIN"
    elif fault == "uncertainty":
        request["uncertainty"].pop("input_quality")
        expected = "ABSTAIN"
    return request, expected


def run(output: Path, trials: int, seed: int) -> dict:
    if output.exists():
        raise ValueError("output is immutable")
    rng = np.random.default_rng(seed)
    base = fixture()
    counts = {fault: {"trials": 0, "correct": 0, "unsafe_valid": 0} for fault in FAULTS}
    samples = []
    for index in range(trials):
        fault = FAULTS[int(rng.integers(len(FAULTS)))]
        request, expected = mutate(base, fault)
        forecast = replay_forecast(**request)
        observed = forecast["forecast_status"]
        unsafe_valid = expected == "ABSTAIN" and observed != "ABSTAIN"
        counts[fault]["trials"] += 1
        counts[fault]["correct"] += int(observed == expected)
        counts[fault]["unsafe_valid"] += int(unsafe_valid)
        if index < 50:
            samples.append({"trial": index, "fault": fault, "expected": expected,
                            "observed": observed, "reasons": forecast["abstention_reasons"]})
    invalid_trials = sum(value["trials"] for key, value in counts.items()
                         if key not in {"none", "optional_missing"})
    unsafe = sum(value["unsafe_valid"] for value in counts.values())
    correct = sum(value["correct"] for value in counts.values())
    # Comparator is a precisely defined unguarded serializer which emits the
    # supplied probability regardless of input/evidence state.
    unguarded_unsafe = invalid_trials
    result = {
        "scope": "SYNTHETIC_SOFTWARE_VALIDITY_BENCHMARK_NOT_SEP_SKILL",
        "seed": seed, "trials": trials, "fault_types": list(FAULTS),
        "iris_status_accuracy": correct / trials,
        "iris_unsafe_valid_outputs": unsafe,
        "iris_unsafe_valid_rate_on_injected_invalid_inputs": unsafe / invalid_trials,
        "unguarded_serializer_unsafe_valid_outputs": unguarded_unsafe,
        "unguarded_serializer_unsafe_valid_rate_on_injected_invalid_inputs": 1.0,
        "counts": counts, "sample_receipts": samples,
        "locked_test_accessed": False, "scientific_superiority_established": False,
        "claim_boundary": "Tests contract enforcement against a defined no-validation wrapper; it is not a comparison with a named SEP competitor."
    }
    output.mkdir(parents=True)
    payload = json.dumps(result, indent=2, allow_nan=False).encode() + b"\n"
    (output / "result.json").write_bytes(payload)
    receipt = {"result_sha256": digest(payload), "runner_sha256": digest(Path(__file__).read_bytes()),
               "trials": trials, "unsafe_valid_outputs": unsafe,
               "locked_test_accessed": False}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.trials, args.seed), indent=2))
