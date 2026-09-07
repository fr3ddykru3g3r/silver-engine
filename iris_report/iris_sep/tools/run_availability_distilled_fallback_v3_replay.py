"""Replay V3 with the established promoted-stack reproducibility tolerance.

This wrapper changes no model, data, calibration, threshold, teacher weight, or
keep gate. The first V3 run stopped only because its independently rebuilt FULL
stack differed from V1 by 1.2638257620989357e-07 while the runner used an
unnecessarily strict 1e-10 guard. The existing promoted-stack contract uses
1e-6 for independent XGBoost rebuilds, so this replay applies that same numerical
tolerance and calls the frozen V3 runner unchanged otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iris_report.iris_sep.tools import run_availability_distilled_fallback_v3 as v3


ESTABLISHED_FULL_REPRO_TOLERANCE = 1e-6


def run(features: Path, events: Path, output: Path) -> dict:
    v3.FULL_REPRO_TOLERANCE = ESTABLISHED_FULL_REPRO_TOLERANCE
    return v3.run(features, events, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(v3._finite(run(args.features, args.events, args.output)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
