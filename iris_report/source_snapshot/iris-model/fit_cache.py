"""Fail-closed planning and validation for the real SHARP FITS cache.

The Colab workflow keeps binary magnetograms outside the source archive.  This
module defines the deterministic subset needed by the current runner and makes
the runner refuse to train when even one required image is absent or malformed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def is_fits_payload(path: str | Path) -> bool:
    """Return a cheap, conservative validity result for a local FITS file."""
    p = Path(path)
    try:
        if p.stat().st_size <= 2880:
            return False
        with p.open("rb") as stream:
            return stream.read(80).startswith(b"SIMPLE")
    except OSError:
        return False


def is_readable_fits(path: str | Path) -> bool:
    """Validate the FITS structure and require a readable 2-D image HDU."""
    if not is_fits_payload(path):
        return False
    try:
        from astropy.io import fits
        import numpy as np
        with fits.open(path, memmap=False, mode="readonly", do_not_scale_image_data=False) as hdul:
            return any(hdu.data is not None and np.ndim(hdu.data) == 2 for hdu in hdul)
    except Exception:
        return False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_temporal_subset(records: pd.DataFrame, per_group: int,
                           positive_slots: int, seed: int) -> pd.DataFrame:
    """Mirror train_generator_v2.group_temporal_subset without importing torch."""
    import numpy as np

    def temporal_even(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or frame.empty:
            return frame.iloc[0:0].copy()
        z = frame.sort_values("t_rec").reset_index(drop=True)
        if len(z) <= n:
            return z.copy()
        idx = np.unique(np.round(np.linspace(0, len(z) - 1, n)).astype(int))
        if len(idx) < n:
            used = set(idx.tolist())
            idx = np.concatenate(
                [idx, np.asarray([i for i in range(len(z)) if i not in used][:n - len(idx)])]
            )
        return z.iloc[np.sort(idx[:n])].copy()

    if per_group <= 0:
        return records.copy().reset_index(drop=True)
    pieces = []
    for _, group in records.groupby("region_group_id", sort=True):
        pos = group[group.label_m1plus_24h.eq(1)]
        neg = group[group.label_m1plus_24h.eq(0)]
        kp = min(max(0, positive_slots), len(pos), per_group)
        kn = min(per_group - kp, len(neg))
        if kp == 0:
            kn = min(per_group, len(neg))
        selected = pd.concat(
            [temporal_even(pos, kp), temporal_even(neg, kn)], ignore_index=True
        )
        if len(selected) < per_group:
            rest = group[~group.sample_id.isin(selected.sample_id)]
            selected = pd.concat(
                [selected, temporal_even(rest, min(per_group - len(selected), len(rest)))],
                ignore_index=True,
            )
        pieces.append(selected)
    if not pieces:
        return records.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _group_subset(records: pd.DataFrame, per_group: int, pos_cap: int,
                  seed: int) -> pd.DataFrame:
    """Mirror train_matched_augmentation.group_subset for planning."""
    import numpy as np

    def temporal_even(frame: pd.DataFrame, n: int) -> pd.DataFrame:
        if n <= 0 or frame.empty:
            return frame.iloc[0:0].copy()
        if len(frame) <= n:
            return frame.copy()
        z = frame.sort_values("t_rec").reset_index(drop=True)
        ids = np.unique(np.round(np.linspace(0, len(z) - 1, n)).astype(int))
        if len(ids) < n:
            used = set(ids.tolist())
            ids = np.r_[ids, [i for i in range(len(z)) if i not in used][:n - len(ids)]]
        return z.iloc[np.asarray(ids[:n], dtype=int)].copy()

    pieces = []
    for _, group in records.groupby("region_group_id", sort=True):
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
                [selected, temporal_even(rest, min(per_group - len(selected), len(rest)))],
                ignore_index=True,
            )
        pieces.append(selected)
    if not pieces:
        return records.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def required_records(evidence_dir: str | Path, *, run_base: bool,
                     run_physics: bool, run_downstream: bool,
                     seed: int = 2026) -> pd.DataFrame:
    """Return the exact image identities needed by the configured runner.

    The generator and downstream scripts each perform their own deterministic
    sampling.  This planner intentionally mirrors those functions so the data
    gate checks the same identities that the training code will request.
    """
    derived = Path(evidence_dir) / "data" / "derived"
    manifest_path = derived / "training_manifest.csv.gz"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing evidence manifest: {manifest_path}")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    parts = {}
    for partition in ("train", "validation", "test"):
        parts[partition] = manifest[
            manifest.partition.eq(partition) & manifest.label_m1plus_24h.notna()
        ].copy()

    frames = []
    if run_base or run_physics:
        generator_train = _group_temporal_subset(parts["train"], 4, 4, seed)
        generator_eval = parts["train"][parts["train"].label_m1plus_24h.eq(1)].copy()
        generator_eval = _group_temporal_subset(generator_eval, 2, 2, seed)
        generator_train = generator_train.assign(required_for="generator_train")
        generator_eval = generator_eval.assign(required_for="generator_eval")
        frames.extend([generator_train, generator_eval])
    if run_downstream:
        downstream_train = _group_subset(parts["train"], 4, 2, seed)
        downstream_val = _group_subset(parts["validation"], 6, 2, seed + 1)
        downstream_test = _group_subset(parts["test"], 6, 2, seed + 2)
        frames.extend([
            downstream_train.assign(required_for="downstream_train"),
            downstream_val.assign(required_for="downstream_validation"),
            downstream_test.assign(required_for="downstream_test"),
        ])
    if not frames:
        raise RuntimeError("No training stage is enabled; cannot plan a FITS cache")
    requested = pd.concat(frames, ignore_index=True)
    requested = requested.drop_duplicates("sample_id").reset_index(drop=True)
    return requested


def index_local_fits(source_dir: str | Path) -> dict[str, Path]:
    """Index ``sample_id.fits`` files, rejecting ambiguous duplicate IDs."""
    root = Path(source_dir)
    if not root.is_dir():
        raise RuntimeError(f"FITS source directory does not exist: {root}")
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in root.rglob("*.fits"):
        sid = path.stem
        if sid in index:
            duplicates.setdefault(sid, [index[sid]]).append(path)
        else:
            index[sid] = path
    if duplicates:
        sample = next(iter(duplicates.items()))
        raise RuntimeError(f"Duplicate FITS sample_id {sample[0]}: {sample[1]}")
    return index


def verify_local_cache(evidence_dir: str | Path, source_dir: str | Path, *,
                       run_base: bool, run_physics: bool,
                       run_downstream: bool, seed: int = 2026,
                       strict: bool = False,
                       write_report: str | Path | None = None) -> dict:
    """Verify all planned files and optionally write a checksum manifest."""
    requested = required_records(
        evidence_dir,
        run_base=run_base,
        run_physics=run_physics,
        run_downstream=run_downstream,
        seed=seed,
    )
    index = index_local_fits(source_dir)
    missing = []
    invalid = []
    rows = []
    for row in requested.itertuples(index=False):
        sid = str(row.sample_id)
        path = index.get(sid)
        if path is None:
            missing.append(sid)
            continue
        valid = is_readable_fits(path) if strict else is_fits_payload(path)
        if not valid:
            invalid.append({"sample_id": sid, "path": str(path)})
            continue
        rows.append({
            "sample_id": sid,
            "fits_path": str(path.resolve()),
            "fits_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "required_for": str(row.required_for),
        })
    report = {
        "status": "PASS" if not missing and not invalid else "FAIL",
        "source_dir": str(Path(source_dir).resolve()),
        "planned_samples": int(len(requested)),
        "valid_samples": int(len(rows)),
        "missing_samples": missing[:50],
        "missing_count": len(missing),
        "invalid_samples": invalid[:50],
        "invalid_count": len(invalid),
        "run_base": bool(run_base),
        "run_physics": bool(run_physics),
        "run_downstream": bool(run_downstream),
        "seed": int(seed),
        "strict_fits_validation": bool(strict),
    }
    if write_report is not None:
        out = Path(write_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        if rows:
            pd.DataFrame(rows).sort_values("sample_id").to_csv(
                out.with_name("fits_cache_manifest.csv.gz"),
                index=False,
                compression="gzip",
            )
    if missing or invalid:
        examples = missing[:5] + [x["sample_id"] for x in invalid[:5]]
        raise RuntimeError(
            f"FITS cache incomplete: planned={len(requested)}, valid={len(rows)}, "
            f"missing={len(missing)}, invalid={len(invalid)}; examples={examples}. "
            "Run colab/acquire_sharp_fits.py in Colab, then rerun this stage."
        )
    return report


def configure_local_source(source_dir: str | Path) -> None:
    """Configure data.py to consume only the verified local cache."""
    os.environ["IRIS_FITS_SOURCE"] = str(Path(source_dir).resolve())
    os.environ["IRIS_REQUIRE_LOCAL_FITS"] = "1"
