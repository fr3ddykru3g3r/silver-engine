#!/usr/bin/env python3
"""Build one predictor-only SEP forecast validity record.

The input assessment must contain only information available no later than the
forecast issue time. This tool never reads SEP outcomes or protected labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iris_sep.forecast_validity import build_validity_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    assessment = json.loads(args.assessment.read_text(encoding="utf-8"))
    if assessment.get("protected_outcomes_accessed") not in (None, False):
        raise ValueError("assessment indicates protected outcome access; refusing to build validity record")
    record = build_validity_record(assessment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "issue_time": record["issue_time"],
        "validity_state": record["validity_state"],
        "reason_codes": record["reason_codes"],
        "validity_record_hash": record["validity_record_hash"],
        "protected_outcomes_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
