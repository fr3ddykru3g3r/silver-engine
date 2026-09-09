"""Replay distilled V3 package export with the established XGBoost tolerance.

The package exporter independently rebuilds the frozen family specialists before
serializing them. Existing IRIS promoted-stack replays established 1e-6 as the
accepted maximum absolute difference for this deterministic-but-not-bit-identical
XGBoost rebuild path. No model parameter, threshold, calibration or V3 keep gate
is changed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from iris_report.iris_sep.tools import export_availability_model_package_v3 as exporter


ESTABLISHED_PACKAGE_REPLAY_TOLERANCE = 1e-6


def run(features: Path, events: Path, output: Path) -> dict[str, object]:
    exporter.REPLAY_TOLERANCE = ESTABLISHED_PACKAGE_REPLAY_TOLERANCE
    return exporter.run(features, events, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.features, args.events, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
