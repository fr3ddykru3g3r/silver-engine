"""Offline tests for the Colab FITS data gate; no network or torch required."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "iris-model"))
from fit_cache import is_fits_payload, verify_local_cache


def write_fake_fits(path: Path) -> None:
    header = b"SIMPLE  =                    T".ljust(2880, b" ")
    path.write_bytes(header + b"0" * 2880)


def make_evidence(root: Path) -> None:
    derived = root / "data" / "derived"
    derived.mkdir(parents=True)
    rows = []
    for i in range(8):
        rows.append({
            "sample_id": f"S{i}", "partition": "train", "label_m1plus_24h": int(i % 2),
            "region_group_id": f"G{i}", "harpnum": i, "t_rec": f"2020-01-01T{i:02d}:00:00+00:00",
        })
    rows += [{
        "sample_id": "V0", "partition": "validation", "label_m1plus_24h": 0,
        "region_group_id": "GV", "harpnum": 100, "t_rec": "2020-02-01T00:00:00+00:00",
    }, {
        "sample_id": "T0", "partition": "test", "label_m1plus_24h": 0,
        "region_group_id": "GT", "harpnum": 101, "t_rec": "2020-03-01T00:00:00+00:00",
    }]
    pd.DataFrame(rows).to_csv(derived / "training_manifest.csv.gz", index=False, compression="gzip")


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        evidence = root / "evidence"
        cache = root / "fits"
        make_evidence(evidence)
        cache.mkdir()
        assert not is_fits_payload(cache / "missing.fits")
        (cache / "S0.fits").write_bytes(b"HTML" * 1000)
        try:
            verify_local_cache(evidence, cache, run_base=True, run_physics=False, run_downstream=False)
        except RuntimeError as exc:
            assert "FITS cache incomplete" in str(exc)
        else:
            raise AssertionError("incomplete cache was accepted")

        for i in range(8):
            write_fake_fits(cache / f"S{i}.fits")
        report = verify_local_cache(
            evidence, cache, run_base=True, run_physics=False, run_downstream=False,
            write_report=root / "report.json",
        )
        assert report["status"] == "PASS"
        assert report["planned_samples"] == 8
        assert json.loads((root / "report.json").read_text())["valid_samples"] == 8
    print("fit cache self-test PASS")


if __name__ == "__main__":
    main()
