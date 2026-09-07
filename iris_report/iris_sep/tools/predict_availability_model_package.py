"""Load-only inference for an exported IRIS-SEP availability model package."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from iris_report.iris_sep.src.iris_sep.model_package import LoadedAvailabilityPackage, STATE_EXPERTS


def run(package: Path, input_csv: Path, output_csv: Path):
    if output_csv.exists():
        raise ValueError("output file must be new")
    model = LoadedAvailabilityPackage.load(package)
    frame = pd.read_csv(input_csv, low_memory=False)
    if "issue_time" not in frame:
        raise ValueError("input requires issue_time")
    out = pd.DataFrame({"issue_time": frame["issue_time"].astype(str)})
    for state in STATE_EXPERTS:
        out[f"p_{state}"] = model.predict(frame, state=state)
        for policy in ("MAX_TSS", "POD80_MIN_FAR"):
            decision = model.decision(frame, state=state, policy=policy)
            out[f"alert_{state}_{policy}"] = decision["alert"].astype(int)
    out.to_csv(output_csv, index=False, float_format="%.17g")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.package, args.input, args.output)


if __name__ == "__main__":
    main()
