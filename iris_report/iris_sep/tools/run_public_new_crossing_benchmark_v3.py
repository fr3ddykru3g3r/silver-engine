"""Serialization-only hardening of the public NEW-crossing benchmark.

The v2 computation reached final receipt serialization before failing because a
legitimately undefined diagnostic metric was represented as NaN and the receipt
writer correctly forbids non-standard JSON NaN literals.  No scientific metric
was printed or inspected before this change.

V3 changes no target, unit, feature, model, seed, calibration, threshold,
bootstrap or score-role behavior.  It only converts non-finite scalar diagnostic
values to JSON null at serialization time.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark_v2 as v2


_ORIGINAL_DUMPS = json.dumps


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_dumps(value, *args, **kwargs):
    return _ORIGINAL_DUMPS(_json_safe(value), *args, **kwargs)


def run(features: Path, events: Path, output: Path):
    # v1.run references its imported json module directly for the final print.
    # Patch only dumps so both receipt writing and stdout remain strict JSON.
    v1.json.dumps = _safe_dumps
    result = v2.run(features, events, output)
    receipt = {
        "status": "SERIALIZATION_ONLY_HARDENING_APPLIED",
        "parent_runner": "run_public_new_crossing_benchmark_v2.py",
        "change": "NONFINITE_DIAGNOSTIC_SCALARS_SERIALIZE_AS_JSON_NULL",
        "undefined_metrics_are_not_coerced_to_zero": True,
        "target_or_model_behavior_changed": False,
        "scientific_score_seen_before_change": False,
        "locked_test_accessed": False,
    }
    Path(output, "v3_serialization_receipt.json").write_text(
        _ORIGINAL_DUMPS(receipt, indent=2, sort_keys=True) + "\n",
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
