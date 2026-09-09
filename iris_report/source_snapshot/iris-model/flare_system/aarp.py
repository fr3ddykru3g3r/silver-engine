from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable

import numpy as np


DATASET_DOI = "doi:10.7910/DVN/WPN39J"
DATASET_VERSION = "1.0"
M1_24H_FILE_ID = 6377544
M1_24H_FILENAME = "M10min_Z00max_..._val24hr.nc"
M1_24H_BYTES = 24_895_924
M1_24H_MD5 = "91da3b78dae5bca2c50601bcb5d4c897"
M1_24H_URL = f"https://dataverse.harvard.edu/api/access/datafile/{M1_24H_FILE_ID}?format=original"

_AARP_ID = re.compile(
    r"^(?P<date>\d{4}\.\d{2}\.\d{2})_(?P<time>\d{2}:\d{2}:\d{2})_"
    r"7h@1h_(?P<aarp>\d+)\.fits$"
)


@dataclass(frozen=True)
class AARPIdentity:
    source_id: str
    issue_time_utc: str
    aarp_number: int


def digest_file(path: str | Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_parameter_file(path: str | Path) -> dict:
    candidate = Path(path)
    if not candidate.is_file():
        raise RuntimeError(f"AARP_PARAMETER_FILE_MISSING: {candidate}")
    size = candidate.stat().st_size
    checksum = digest_file(candidate, "md5")
    if size != M1_24H_BYTES or checksum != M1_24H_MD5:
        raise RuntimeError(
            "AARP_PARAMETER_FILE_INVALID: "
            f"bytes={size} expected_bytes={M1_24H_BYTES} "
            f"md5={checksum} expected_md5={M1_24H_MD5}"
        )
    return {
        "status": "PASS",
        "dataset_doi": DATASET_DOI,
        "dataset_version": DATASET_VERSION,
        "file_id": M1_24H_FILE_ID,
        "bytes": size,
        "md5": checksum,
        "sha256": digest_file(candidate, "sha256"),
    }


def decode_fixed_width(values: np.ndarray) -> list[str]:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected a two-dimensional character array, got {array.shape}")
    return [
        b"".join(np.asarray(row).tolist()).decode("utf-8", "strict").rstrip("\x00 ").strip()
        for row in array
    ]


def parse_aarp_id(value: str) -> AARPIdentity:
    match = _AARP_ID.fullmatch(str(value))
    if match is None:
        raise ValueError(f"Unrecognized AARP identity: {value!r}")
    timestamp = f"{match.group('date').replace('.', '-') }T{match.group('time')}Z"
    return AARPIdentity(str(value), timestamp, int(match.group("aarp")))


def _counts(values: Iterable[int]) -> dict[str, int]:
    unique, counts = np.unique(np.asarray(list(values), dtype=int), return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


def audit_parameter_file(path: str | Path) -> dict:
    """Verify and summarize the published M1+/24 h AARP parameter matrix.

    This intentionally does not join AARP numbers to HARP numbers. They are
    different source namespaces until an explicit, independently verified map
    is supplied; numeric equality is not accepted as an identity bridge.
    """
    provenance = verify_parameter_file(path)
    try:
        from scipy.io import netcdf_file
    except ImportError as exc:  # pragma: no cover - dependency error is explicit
        raise RuntimeError("scipy is required to inspect the AARP NetCDF") from exc

    with netcdf_file(path, "r", mmap=False) as dataset:
        required = {"vars", "pops", "groups", "idlist", "varnames"}
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise RuntimeError(f"AARP_SCHEMA_INVALID: missing variables {missing}")
        values = np.asarray(dataset.variables["vars"].data).copy()
        populations = np.asarray(dataset.variables["pops"].data).astype(int, copy=True)
        groups = np.asarray(dataset.variables["groups"].data).astype(int, copy=True)
        identities = decode_fixed_width(dataset.variables["idlist"].data)
        variable_names = decode_fixed_width(dataset.variables["varnames"].data)

    expected_shape = (176, 32_067)
    if values.shape != expected_shape:
        raise RuntimeError(f"AARP_SCHEMA_INVALID: vars shape {values.shape}, expected {expected_shape}")
    if len(identities) != expected_shape[1] or len(variable_names) != expected_shape[0]:
        raise RuntimeError("AARP_SCHEMA_INVALID: identity or variable-name dimension mismatch")
    parsed = [parse_aarp_id(value) for value in identities]
    if len({item.source_id for item in parsed}) != len(parsed):
        raise RuntimeError("AARP_SCHEMA_INVALID: duplicate source identities")

    nonfinite = int(values.size - np.isfinite(values).sum())
    return {
        **provenance,
        "schema": {
            "variables": int(values.shape[0]),
            "examples": int(values.shape[1]),
            "finite_values": int(np.isfinite(values).sum()),
            "nonfinite_values": nonfinite,
            "population_counts": _counts(populations),
            "group_counts": _counts(groups),
            "unique_aarp_numbers": len({item.aarp_number for item in parsed}),
            "first_issue_time_utc": min(item.issue_time_utc for item in parsed),
            "last_issue_time_utc": max(item.issue_time_utc for item in parsed),
            "first_variables": variable_names[:12],
        },
        "identity_bridge": {
            "status": "REQUIRED_BEFORE_FUSION",
            "reason": "AARP numbers are not assumed to equal JSOC HARPNUM values",
            "accepted_key": "verified mapping plus issue time",
        },
        "recommended_use": [
            "external NCI/NPDA comparator",
            "AIA-only temporal-feature expert",
            "late fusion with HMI only after a leakage-safe identity bridge",
        ],
    }
