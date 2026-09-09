"""Isolated receipt-serialization hardening for public NEW-crossing benchmark.

V3 changed the process-global ``json.dumps`` function through the shared json
module object. XGBoost also uses that module internally, so its NaN missing-value
sentinel was serialized as JSON null and rejected before model fitting.

V4 keeps the v2 scientific method exactly unchanged and replaces only the
``json`` object referenced by the v1 runner with a tiny proxy. XGBoost and every
other dependency retain the standard-library json module unchanged.

No scientific score was exposed before this correction.
"""
from __future__ import annotations

import argparse
import json as std_json
import math
from pathlib import Path

import numpy as np

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as v1
from iris_report.iris_sep.tools import run_public_new_crossing_benchmark_v2 as v2


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


class _ReceiptJsonProxy:
    @staticmethod
    def dumps(value, *args, **kwargs):
        return std_json.dumps(_json_safe(value), *args, **kwargs)

    @staticmethod
    def loads(value, *args, **kwargs):
        return std_json.loads(value, *args, **kwargs)


def run(features: Path, events: Path, output: Path):
    # Replace only the runner module's reference; never mutate std_json.dumps.
    v1.json = _ReceiptJsonProxy
    result = v2.run(features, events, output)
    receipt = {
        "status": "ISOLATED_SERIALIZATION_HARDENING_APPLIED",
        "parent_scientific_runner": "run_public_new_crossing_benchmark_v2.py",
        "change": "RUNNER_LOCAL_JSON_PROXY_CONVERTS_NONFINITE_RECEIPT_SCALARS_TO_NULL",
        "xgboost_json_module_modified": False,
        "undefined_metrics_are_not_coerced_to_zero": True,
        "target_or_model_behavior_changed": False,
        "scientific_score_seen_before_change": False,
        "locked_test_accessed": False,
    }
    Path(output, "v4_serialization_receipt.json").write_text(
        std_json.dumps(receipt, indent=2, sort_keys=True) + "\n",
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
