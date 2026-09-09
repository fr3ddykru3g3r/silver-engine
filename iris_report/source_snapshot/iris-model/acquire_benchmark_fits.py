#!/usr/bin/env python3
"""Acquire the independent benchmark's predeclared, label-blind FITS identities."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pandas as pd

from flare_system.data import SamplingConfig, build_selected_records


def _load_acquirer(path: Path):
    spec = importlib.util.spec_from_file_location("iris_jsoc_acquirer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load acquisition module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_targets(evidence_dir: Path, seed: int) -> pd.DataFrame:
    frames = build_selected_records(evidence_dir, SamplingConfig(seed=seed))
    targets = pd.concat(frames.values(), ignore_index=True).drop_duplicates("sample_id")
    targets = targets.copy()
    targets["tai_trec"] = targets["T_REC"].astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--acquisition-script", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    evidence_dir = Path(args.evidence_dir)
    acquirer = _load_acquirer(Path(args.acquisition_script))
    frozen = benchmark_targets(evidence_dir, args.seed)
    acquirer.load_targets = lambda _evidence, _scope, _seed: frozen.copy()
    report = acquirer.acquire(
        evidence_dir,
        Path(args.output_dir),
        "all",
        args.seed,
        os.environ.get("JSOC_EMAIL", "").strip(),
        batch_size=args.batch_size,
        skip_verify=True,
    )
    print({"benchmark_planned_samples": len(frozen), **report}, flush=True)


if __name__ == "__main__":
    main()
