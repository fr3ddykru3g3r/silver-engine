"""Acquire and verify the complete FITS set required by the locked downstream matrix.

The existing BASE cache is used as a seed.  Only downstream records absent from
that seed are fetched from JSOC, and the final directory is checked against the
same deterministic sampling plan used by the downstream trainer.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from data import _download_one, build_records
from fit_cache import index_local_fits, is_fits_payload, verify_local_cache


def temporal_even(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    if len(frame) <= n:
        return frame.copy()
    z = frame.sort_values("t_rec").reset_index(drop=True)
    ids = np.unique(np.round(np.linspace(0, len(z) - 1, n)).astype(int))
    if len(ids) < n:
        used = set(ids.tolist())
        ids = np.r_[ids, [i for i in range(len(z)) if i not in used][: n - len(ids)]]
    return z.iloc[np.asarray(ids[:n], dtype=int)].copy()


def group_subset(frame: pd.DataFrame, per_group: int, pos_cap: int, seed: int) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("region_group_id", sort=True):
        pos = group[group.label_m1plus_24h.eq(1)]
        neg = group[group.label_m1plus_24h.eq(0)]
        kp = min(pos_cap, len(pos), per_group)
        kn = min(per_group - kp, len(neg))
        if kp == 0:
            kn = min(per_group, len(neg))
        selected = pd.concat(
            [temporal_even(pos, kp), temporal_even(neg, kn)], ignore_index=True
        )
        if len(selected) < per_group:
            rest = group[~group.sample_id.isin(selected.sample_id)]
            selected = pd.concat(
                [
                    selected,
                    temporal_even(rest, min(per_group - len(selected), len(rest))),
                ],
                ignore_index=True,
            )
        pieces.append(selected)
    if not pieces:
        return frame.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def downstream_plan(evidence_dir: str | Path, seed: int) -> pd.DataFrame:
    frames = [
        group_subset(build_records(evidence_dir, "train"), 4, 2, seed),
        group_subset(build_records(evidence_dir, "validation"), 6, 2, seed + 1),
        group_subset(build_records(evidence_dir, "test"), 6, 2, seed + 2),
    ]
    return pd.concat(frames, ignore_index=True).drop_duplicates("sample_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    requested = downstream_plan(args.evidence_dir, args.seed)
    requested.to_csv(output / "complete_requested_manifest.csv.gz", index=False, compression="gzip")

    existing = index_local_fits(output)
    pending = [
        row
        for row in requested.to_dict("records")
        if not (existing.get(str(row["sample_id"])) and is_fits_payload(existing[str(row["sample_id"])]))
    ]
    print(
        json.dumps(
            {"planned": len(requested), "already_valid": len(requested) - len(pending), "pending": len(pending)},
            sort_keys=True,
        ),
        flush=True,
    )

    failures = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, min(int(args.workers), 8))) as pool:
            futures = {
                pool.submit(_download_one, row, output, 12): row for row in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append({"sample_id": str(row["sample_id"]), "error": str(exc)})
                if index % 100 == 0 or index == len(pending):
                    print(
                        f"acquired {index}/{len(pending)}; failures={len(failures)}",
                        flush=True,
                    )
    if failures:
        raise RuntimeError(
            f"{len(failures)} downstream FITS records failed; examples={failures[:5]}"
        )

    report = verify_local_cache(
        args.evidence_dir,
        output,
        run_base=False,
        run_physics=False,
        run_downstream=True,
        seed=args.seed,
        strict=True,
        write_report=output / "complete_fits_cache_report.json",
    )
    report.update({"planned_by_downstream_script": int(len(requested)), "pending_downloads": int(len(pending))})
    (output / "complete_fits_acquisition_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
