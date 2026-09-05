from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Allow split/provenance tests on lightweight audit hosts.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from fit_cache import is_readable_fits, sha256_file


@dataclass(frozen=True)
class TensorPreprocessConfig:
    output_size: int = 128
    fov_mm: float = 256.0
    clip_gauss: float = 3000.0
    asinh_scale_gauss: float = 250.0
    centroid_threshold_gauss: float = 100.0


@dataclass(frozen=True)
class SamplingConfig:
    train_per_group: int = 4
    validation_per_group: int = 6
    test_per_group: int = 6
    positive_cap: int = 2
    seed: int = 2026
    temporal_buffer_hours: float = 36.0


def _temporal_even(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    ordered = frame.sort_values(["t_rec", "sample_id"]).reset_index(drop=True)
    if len(ordered) <= n:
        return ordered.copy()
    indices = np.unique(np.round(np.linspace(0, len(ordered) - 1, n)).astype(int))
    if len(indices) < n:
        used = set(indices.tolist())
        indices = np.r_[indices, [i for i in range(len(ordered)) if i not in used][: n - len(indices)]]
    return ordered.iloc[np.sort(indices[:n])].copy()


def select_group_subset(frame: pd.DataFrame, per_group: int, positive_cap: int, seed: int) -> pd.DataFrame:
    """Select a deterministic, temporally spread, group-balanced subset."""
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("region_group_id", sort=True):
        positive = group[group.label_m1plus_24h.eq(1)]
        negative = group[group.label_m1plus_24h.eq(0)]
        n_positive = min(positive_cap, len(positive), per_group)
        n_negative = min(per_group - n_positive, len(negative))
        if n_positive == 0:
            n_negative = min(per_group, len(negative))
        selected = pd.concat(
            [_temporal_even(positive, n_positive), _temporal_even(negative, n_negative)],
            ignore_index=True,
        )
        if len(selected) < per_group:
            remaining = group[~group.sample_id.isin(selected.sample_id)]
            selected = pd.concat(
                [selected, _temporal_even(remaining, min(per_group - len(selected), len(remaining)))],
                ignore_index=True,
            )
        pieces.append(selected)
    if not pieces:
        return frame.iloc[0:0].copy().reset_index(drop=True)
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def select_label_blind_group_subset(frame: pd.DataFrame, per_group: int, seed: int) -> pd.DataFrame:
    """Predeclare a temporally spread cohort without consulting outcomes."""
    pieces = [
        _temporal_even(group, min(per_group, len(group)))
        for _, group in frame.groupby("region_group_id", sort=True)
    ]
    if not pieces:
        return frame.iloc[0:0].copy().reset_index(drop=True)
    return pd.concat(pieces, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def validate_evidence(evidence_dir: str | Path, temporal_buffer_hours: float = 36.0) -> dict:
    evidence = Path(evidence_dir)
    derived = evidence / "data" / "derived"
    receipt_path = derived / "tai_repair_audit.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "PASS":
        raise RuntimeError(f"TAI repair is not PASS: {receipt_path}")
    manifest = pd.read_csv(derived / "training_manifest.csv.gz", low_memory=False)
    partitions = {}
    for name in ("train", "validation", "test"):
        part = manifest[manifest.partition.eq(name) & manifest.label_m1plus_24h.notna()].copy()
        if part.empty or part.label_m1plus_24h.nunique() != 2:
            raise RuntimeError(f"Partition {name} is empty or lacks both labels")
        partitions[name] = part
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(partitions[left].region_group_id.astype(str)) & set(partitions[right].region_group_id.astype(str))
        if overlap:
            raise RuntimeError(f"Region-group leakage between {left} and {right}: {len(overlap)}")
    times = {name: pd.to_datetime(frame.t_rec, utc=True, errors="raise") for name, frame in partitions.items()}
    buffer = pd.Timedelta(hours=float(temporal_buffer_hours))
    if not (times["train"].max() + buffer < times["validation"].min()):
        raise RuntimeError("Train/validation temporal buffer failed")
    if not (times["validation"].max() + buffer < times["test"].min()):
        raise RuntimeError("Validation/test temporal buffer failed")
    return {
        "status": "PASS",
        "receipt": str(receipt_path),
        "rows": {name: int(len(frame)) for name, frame in partitions.items()},
        "groups": {name: int(frame.region_group_id.nunique()) for name, frame in partitions.items()},
        "positive_rows": {name: int(frame.label_m1plus_24h.sum()) for name, frame in partitions.items()},
        "temporal_buffer_hours": float(temporal_buffer_hours),
        "time_ranges": {
            name: {"min": str(times[name].min()), "max": str(times[name].max())}
            for name in partitions
        },
    }


def build_selected_records(evidence_dir: str | Path, cfg: SamplingConfig) -> dict[str, pd.DataFrame]:
    validate_evidence(evidence_dir, cfg.temporal_buffer_hours)
    full = {name: build_exact_records(evidence_dir, name) for name in ("train", "validation", "test")}
    selected = {
        "train": select_group_subset(full["train"], cfg.train_per_group, cfg.positive_cap, cfg.seed),
        "validation": select_group_subset(full["validation"], cfg.validation_per_group, cfg.positive_cap, cfg.seed + 1),
        # Locked test identities are selected without inspecting outcomes.
        "test": select_label_blind_group_subset(full["test"], cfg.test_per_group, cfg.seed + 2),
    }
    identities: set[str] = set()
    for name, frame in selected.items():
        ids = set(frame.sample_id.astype(str))
        if identities & ids:
            raise RuntimeError(f"Sample identity leakage involving {name}")
        identities |= ids
    return selected


def build_exact_records(evidence_dir: str | Path, partition: str) -> pd.DataFrame:
    """Join labels to geometry by exact HARP and immutable JSOC TAI record key."""
    derived = Path(evidence_dir) / "data" / "derived"
    manifest = pd.read_csv(derived / "training_manifest.csv.gz", low_memory=False)
    metadata = pd.read_csv(derived / "sharp_metadata.csv.gz", low_memory=False)
    manifest = manifest[
        manifest.partition.eq(partition) & manifest.label_m1plus_24h.notna()
    ].copy()
    manifest["harpnum"] = pd.to_numeric(manifest.harpnum, errors="raise").astype(int)
    if "t_rec_tai" not in manifest:
        raise RuntimeError("Manifest has no exact t_rec_tai key")
    normalize_tai = lambda value: re.sub(r"\s+", "", str(value).strip()).upper()
    manifest["tai_key"] = manifest.t_rec_tai.map(normalize_tai)
    metadata["tai_key"] = metadata.T_REC.map(normalize_tai)
    metadata["harpnum"] = pd.to_numeric(metadata.HARPNUM, errors="coerce")
    geometry = metadata[["harpnum", "tai_key", "T_REC", "CDELT1", "CDELT2", "RSUN_REF"]].dropna(
        subset=["harpnum", "tai_key"]
    )
    geometry["harpnum"] = geometry.harpnum.astype(int)
    geometry = geometry.drop_duplicates(["harpnum", "tai_key"])
    records = manifest.merge(
        geometry,
        on=["harpnum", "tai_key"],
        how="left",
        validate="many_to_one",
    )
    required = ["magnetogram_url", "T_REC", "CDELT1", "CDELT2", "RSUN_REF"]
    missing = records[required].isna().any(axis=1)
    if bool(missing.any()):
        examples = records.loc[missing, ["sample_id", "harpnum", "t_rec"]].head().to_dict("records")
        raise RuntimeError(
            f"Exact geometry join failed for {int(missing.sum())} {partition} rows; examples={examples}"
        )
    return records.drop(columns=["tai_key"]).reset_index(drop=True)


def split_validation_roles(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Split validation groups into disjoint monitor, calibration, and threshold roles.

    The assignment depends only on the group identity and seed, never on labels or
    predictions.  This prevents one validation sample from serving three model-
    selection purposes.
    """
    groups = sorted(frame.region_group_id.astype(str).unique())
    ranked = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
    )
    roles = ("validation_monitor", "validation_calibration", "validation_threshold")
    assignments = {group: roles[index % len(roles)] for index, group in enumerate(ranked)}
    out = {
        role: frame[frame.region_group_id.astype(str).map(assignments).eq(role)].copy().reset_index(drop=True)
        for role in roles
    }
    for role, subset in out.items():
        if subset.empty or subset.label_m1plus_24h.nunique() != 2:
            raise RuntimeError(f"Validation role {role} is empty or lacks both labels")
    return out


def attach_verified_fits(frames: dict[str, pd.DataFrame], fits_dir: str | Path) -> dict[str, pd.DataFrame]:
    fits_root = Path(fits_dir)
    # The cache contract is flat (`sample_id.fits`). Recursive Drive walks are
    # both unnecessary and extremely slow on Colab's mounted filesystem.
    index = {path.stem: path for path in fits_root.glob("*.fits")}
    trusted_manifest: dict[str, tuple[int, str]] = {}
    report_path = fits_root / "acquisition_report.json"
    manifest_path = fits_root / "fits_cache_manifest.csv.gz"
    if report_path.is_file() and manifest_path.is_file():
        report = json.loads(report_path.read_text())
        if report.get("status") == "PASS" and report.get("scope") in {"all", "downstream"}:
            manifest = pd.read_csv(manifest_path)
            if {"sample_id", "fits_bytes", "sha256"}.issubset(manifest.columns):
                trusted_manifest = {
                    str(row.sample_id): (int(row.fits_bytes), str(row.sha256))
                    for row in manifest.itertuples(index=False)
                }
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    invalid: list[str] = []
    for name, frame in frames.items():
        paths: list[str | None] = []
        sizes: list[int | None] = []
        hashes: list[str | None] = []
        for sample_id in frame.sample_id.astype(str):
            path = index.get(sample_id)
            if path is None:
                missing.append(sample_id)
                paths.append(None)
                sizes.append(None)
                hashes.append(None)
            elif sample_id not in trusted_manifest and not is_readable_fits(path):
                invalid.append(sample_id)
                paths.append(None)
                sizes.append(None)
                hashes.append(None)
            else:
                paths.append(str(path.resolve()))
                if sample_id in trusted_manifest:
                    expected_size, expected_hash = trusted_manifest[sample_id]
                    actual_hash = sha256_file(path)
                    if path.stat().st_size != expected_size or actual_hash != expected_hash:
                        invalid.append(sample_id)
                        paths[-1] = None
                        sizes.append(None)
                        hashes.append(None)
                    else:
                        sizes.append(expected_size)
                        hashes.append(expected_hash)
                else:
                    sizes.append(int(path.stat().st_size))
                    hashes.append(sha256_file(path))
        z = frame.copy()
        z["fits_path"] = paths
        z["fits_bytes"] = sizes
        z["fits_sha256"] = hashes
        out[name] = z
    if missing or invalid:
        raise RuntimeError(
            f"FITS_CONTRACT_FAILED: missing={len(missing)} invalid={len(invalid)} "
            f"examples={(missing + invalid)[:8]}"
        )
    return out


def _raw_physics_features(frame: pd.DataFrame) -> np.ndarray:
    usflux = pd.to_numeric(frame.usflux, errors="coerce").fillna(0.0).to_numpy(float)
    r_value = pd.to_numeric(frame.r_value, errors="coerce").fillna(0.0).to_numpy(float)
    latitude = pd.to_numeric(frame.latitude_deg, errors="coerce").fillna(0.0).to_numpy(float)
    cmd = pd.to_numeric(frame.cmd_deg, errors="coerce").fillna(0.0).to_numpy(float)
    return np.column_stack(
        [
            np.log1p(np.clip(np.abs(usflux), 0, None)),
            np.log1p(np.clip(np.abs(r_value), 0, None)),
            np.sin(np.deg2rad(np.clip(latitude, -90, 90))),
            np.clip(cmd / 90.0, -2.0, 2.0),
        ]
    ).astype(np.float32)


@dataclass(frozen=True)
class FeatureScaler:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    names: tuple[str, ...] = ("log_usflux", "log_r_value", "sin_latitude", "cmd_fraction")

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FeatureScaler":
        x = _raw_physics_features(frame)
        center = np.nanmedian(x, axis=0)
        q25, q75 = np.nanpercentile(x, [25, 75], axis=0)
        scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
        return cls(tuple(center.tolist()), tuple(scale.tolist()))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x = _raw_physics_features(frame)
        return np.clip((x - np.asarray(self.center)) / np.asarray(self.scale), -8.0, 8.0).astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


def records_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "sample_id", "region_group_id", "t_rec", "label_m1plus_24h",
        "fits_bytes", "fits_sha256", "CDELT1", "CDELT2", "RSUN_REF",
    ]
    fields = frame[[column for column in columns if column in frame]].copy()
    payload = fields.sort_values("sample_id").to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


class TensorCacheDataset(Dataset):
    """Preprocess every FITS file once, then reuse an atomic compressed tensor cache."""

    def __init__(
        self,
        records: pd.DataFrame,
        scaler: FeatureScaler,
        cache_dir: str | Path,
        preprocess: TensorPreprocessConfig = TensorPreprocessConfig(),
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.features = scaler.transform(self.records)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.preprocess = preprocess
        cfg_payload = json.dumps(
            {"preprocess": asdict(preprocess), "schema": 2, "implementation": "pil-aware-v1"},
            sort_keys=True,
        ).encode()
        self.cache_version = hashlib.sha256(cfg_payload).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, row: pd.Series) -> Path:
        identity = json.dumps(
            {
                "sample_id": str(row.sample_id),
                "fits_sha256": str(row.fits_sha256),
                "geometry": [float(row.CDELT1), float(row.CDELT2), float(row.RSUN_REF)],
                "cache_version": self.cache_version,
            },
            sort_keys=True,
        ).encode()
        key = hashlib.sha256(identity).hexdigest()[:24]
        return self.cache_dir / self.cache_version / f"{row.sample_id}-{key}.npz"

    def _load_or_create(self, row: pd.Series) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required to materialize the tensor cache")
        from preprocess import PreprocessConfig, preprocess_fits

        path = self._cache_path(row)
        if path.is_file():
            try:
                with np.load(path) as payload:
                    x = payload["x"].astype(np.float32, copy=False)
                if x.shape == (1, self.preprocess.output_size, self.preprocess.output_size):
                    return x
            except Exception:
                pass
        x, _ = preprocess_fits(
            row.fits_path,
            float(row.CDELT1),
            float(row.CDELT2),
            float(row.RSUN_REF),
            PreprocessConfig(**asdict(self.preprocess)),
        )
        array = x.numpy().astype(np.float32, copy=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, x=array)
        os.replace(temporary, path)
        return array

    def __getitem__(self, index: int) -> dict:
        if torch is None:
            raise RuntimeError("PyTorch is required to load training tensors")
        row = self.records.iloc[index]
        x = self._load_or_create(row)
        physics = self.features[index]
        return {
            "x": torch.from_numpy(x),
            "physics": torch.from_numpy(physics.copy()),
            "aux_target": torch.from_numpy(physics[:2].copy()),
            "y": torch.tensor(float(row.label_m1plus_24h), dtype=torch.float32),
            "sample_id": str(row.sample_id),
            "group": str(row.region_group_id),
        }


def collate(batch: Iterable[dict]) -> dict:
    if torch is None:
        raise RuntimeError("PyTorch is required to collate training tensors")
    rows = list(batch)
    return {
        "x": torch.stack([row["x"] for row in rows]),
        "physics": torch.stack([row["physics"] for row in rows]),
        "aux_target": torch.stack([row["aux_target"] for row in rows]),
        "y": torch.stack([row["y"] for row in rows]),
        "sample_id": [row["sample_id"] for row in rows],
        "group": [row["group"] for row in rows],
    }
