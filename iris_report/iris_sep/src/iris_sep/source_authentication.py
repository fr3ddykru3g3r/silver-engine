"""Trusted-source authentication for prospective IRIS-SEP acquisitions.

This layer is intentionally separate from feature-lineage semantics. A lineage
record can say *what* a value represents; this module verifies that the bound
acquisition actually came from a predeclared official provider/endpoint and was
retrieved no later than the forecast issue time.

Historical backfill remains useful research evidence, but it cannot be relabelled
as prospective availability simply because it was downloaded from an official
archive later.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


FORMAT = "IRIS_SEP_TRUSTED_SOURCE_AUTHENTICATION_V1"
REGISTRY_PATH = Path(__file__).resolve().parents[2] / "config" / "trusted_source_registry_v1.json"


class SourceAuthenticationError(ValueError):
    """Raised when an acquisition receipt is not trusted for prospective use."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourceAuthenticationError("source-authentication payload is not canonical JSON") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _parse_time(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise SourceAuthenticationError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SourceAuthenticationError(f"{name} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SourceAuthenticationError(f"{name} must be timezone-aware")
    return parsed


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAuthenticationError("trusted source registry could not be loaded") from exc
    if not isinstance(payload, dict) or payload.get("registry_id") != "iris-sep-trusted-source-registry-v1":
        raise SourceAuthenticationError("unsupported trusted source registry")
    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise SourceAuthenticationError("trusted source registry has no sources")
    return payload


def registry_sha256(path: Path = REGISTRY_PATH) -> str:
    return sha256_file(path)


def build_acquisition_receipt(
    *,
    source_id: str,
    retrieved_at: datetime,
    artifact_sha256: str,
    artifact_bytes: int,
    source_url: str | None = None,
    query_identity: str | None = None,
    observation_first_utc: str | None = None,
    observation_last_utc: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Build a receipt bound to the checked-in registry.

    This does not itself grant prospective permission; use
    :func:`authenticate_acquisition_receipts` with the actual forecast issue time.
    """
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise SourceAuthenticationError("retrieved_at must be timezone-aware")
    if not _is_sha256(artifact_sha256):
        raise SourceAuthenticationError("artifact_sha256 must be lowercase SHA-256")
    if not isinstance(artifact_bytes, int) or artifact_bytes < 0:
        raise SourceAuthenticationError("artifact_bytes must be a nonnegative integer")
    registry = load_registry(registry_path)
    if source_id not in registry["sources"]:
        raise SourceAuthenticationError("source_id absent from trusted registry")
    receipt = {
        "source_id": str(source_id),
        "registry_id": registry["registry_id"],
        "registry_sha256": registry_sha256(registry_path),
        "retrieved_at_utc": retrieved_at.isoformat(),
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact_bytes,
        "source_url": source_url,
        "query_identity": query_identity,
        "observation_first_utc": observation_first_utc,
        "observation_last_utc": observation_last_utc,
        "metadata": dict(metadata or {}),
    }
    receipt["receipt_sha256"] = sha256_bytes(_canonical_json(receipt))
    return receipt


def _registered_host(source: Mapping[str, Any]) -> str:
    value = source.get("host")
    if not isinstance(value, str) or not value:
        raise SourceAuthenticationError("registry source host missing")
    return value.lower()


def _url_host(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    parsed = urlparse(value)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        return None
    return parsed.hostname.lower()


def _source_url_host_reasons(source_url: Any, source: Mapping[str, Any]) -> list[str]:
    host = _url_host(source_url)
    if host is None:
        return ["HTTPS_SOURCE_URL_REQUIRED"]
    if host != _registered_host(source):
        return ["SOURCE_HOST_MISMATCH"]
    return []


def _matches_endpoint_template(source_url: str, template: str) -> bool:
    # Registry templates intentionally expose only YYYY/MM placeholders. Convert
    # them to a strict full-URL regex; no arbitrary template execution occurs.
    pattern = re.escape(template)
    pattern = pattern.replace(re.escape("{YYYY}"), r"\d{4}")
    pattern = pattern.replace(re.escape("{MM}"), r"(?:0[1-9]|1[0-2])")
    return re.fullmatch(pattern, source_url) is not None


def _receipt_reasons(
    receipt: Mapping[str, Any],
    *,
    issue_time: datetime,
    registry: Mapping[str, Any],
    registry_digest: str,
) -> list[str]:
    reasons: list[str] = []
    source_id = receipt.get("source_id")
    sources = registry["sources"]
    if not isinstance(source_id, str) or source_id not in sources:
        return ["SOURCE_ID_NOT_REGISTERED"]
    source = sources[source_id]

    if receipt.get("registry_id") != registry.get("registry_id"):
        reasons.append("REGISTRY_ID_MISMATCH")
    if receipt.get("registry_sha256") != registry_digest:
        reasons.append("REGISTRY_DIGEST_MISMATCH")
    if not _is_sha256(receipt.get("artifact_sha256")):
        reasons.append("ARTIFACT_DIGEST_INVALID")
    if not _is_sha256(receipt.get("receipt_sha256")):
        reasons.append("RECEIPT_DIGEST_INVALID")
    else:
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_sha256")
        if sha256_bytes(_canonical_json(unsigned)) != claimed:
            reasons.append("RECEIPT_DIGEST_MISMATCH")

    try:
        retrieved = _parse_time(receipt.get("retrieved_at_utc"), "retrieved_at_utc")
    except SourceAuthenticationError:
        retrieved = None
        reasons.append("RETRIEVAL_TIME_INVALID")
    if retrieved is not None and retrieved > issue_time:
        reasons.append("RETRIEVED_AFTER_FORECAST_ISSUE")

    transport = str(source.get("transport", ""))
    source_url = receipt.get("source_url")
    # Every registered remote acquisition must bind the provider host, including
    # higher-level clients such as DRMS and SunPy/HEK.
    reasons.extend(_source_url_host_reasons(source_url, source))

    if transport.startswith("HTTPS"):
        endpoint = source.get("endpoint")
        template = source.get("endpoint_template")
        if isinstance(endpoint, str) and endpoint and source_url != endpoint:
            reasons.append("SOURCE_ENDPOINT_MISMATCH")
        if isinstance(template, str) and template:
            if not isinstance(source_url, str) or not _matches_endpoint_template(source_url, template):
                reasons.append("SOURCE_ENDPOINT_TEMPLATE_MISMATCH")
        if not endpoint and not template:
            reasons.append("REGISTRY_ENDPOINT_RULE_MISSING")
    elif transport == "DRMS_QUERY":
        query = receipt.get("query_identity")
        series = source.get("series")
        if not isinstance(query, str) or not isinstance(series, str) or series not in query:
            reasons.append("DRMS_SERIES_NOT_BOUND_IN_QUERY")
    elif transport == "SUNPY_FIDO_HEK":
        query = receipt.get("query_identity")
        expected = source.get("query_identity")
        if query != expected:
            reasons.append("HEK_QUERY_IDENTITY_MISMATCH")
    else:
        reasons.append("UNSUPPORTED_SOURCE_TRANSPORT")

    if source.get("prospective_admissibility") not in {
        "SNAPSHOT_BEFORE_ISSUE_ONLY",
        "QUERY_COMPLETED_BEFORE_ISSUE_ONLY",
    }:
        reasons.append("SOURCE_NOT_REGISTERED_FOR_PROSPECTIVE_USE")
    return reasons


def authenticate_acquisition_receipts(
    *,
    issue_time: datetime,
    receipts: Sequence[Mapping[str, Any]],
    required_source_ids: Sequence[str] | None = None,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    """Authenticate a set of exact acquisition receipts and fail closed.

    A historical archive downloaded after ``issue_time`` is rejected for
    prospective use even when all hashes and provider identities are valid.
    """
    if issue_time.tzinfo is None or issue_time.utcoffset() is None:
        raise SourceAuthenticationError("issue_time must be timezone-aware")
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise SourceAuthenticationError("receipts must be a sequence")
    registry = load_registry(registry_path)
    digest = registry_sha256(registry_path)

    by_id: dict[str, Mapping[str, Any]] = {}
    structural: list[str] = []
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, Mapping):
            structural.append(f"receipt[{index}]:NOT_A_MAPPING")
            continue
        source_id = str(receipt.get("source_id", ""))
        if not source_id:
            structural.append(f"receipt[{index}]:SOURCE_ID_MISSING")
            continue
        if source_id in by_id:
            structural.append(f"{source_id}:DUPLICATE_RECEIPT")
            continue
        by_id[source_id] = receipt

    required = tuple(str(value) for value in (required_source_ids or by_id.keys()))
    if any(not value for value in required) or len(set(required)) != len(required):
        raise SourceAuthenticationError("required_source_ids must contain unique non-empty values")
    for source_id in required:
        if source_id not in by_id:
            structural.append(f"{source_id}:MISSING_RECEIPT")

    rows = []
    reasons = list(structural)
    for source_id in sorted(set(required)):
        receipt = by_id.get(source_id)
        if receipt is None:
            rows.append({"source_id": source_id, "authenticated": False, "reasons": ["MISSING_RECEIPT"]})
            continue
        local = _receipt_reasons(receipt, issue_time=issue_time, registry=registry, registry_digest=digest)
        reasons.extend(f"{source_id}:{reason}" for reason in local)
        rows.append({"source_id": source_id, "authenticated": not local, "reasons": local})

    authenticated = not reasons and all(row["authenticated"] for row in rows)
    payload = {
        "format": FORMAT,
        "registry_id": registry["registry_id"],
        "registry_sha256": digest,
        "issue_time": issue_time.isoformat(),
        "required_source_ids": list(required),
        "source_results": rows,
        "authenticated_for_prospective_use": authenticated,
        "reasons": sorted(set(reasons)),
    }
    payload["authentication_receipt_sha256"] = sha256_bytes(_canonical_json(payload))
    return payload


def require_authenticated_acquisitions(**kwargs: Any) -> dict[str, Any]:
    result = authenticate_acquisition_receipts(**kwargs)
    if not result["authenticated_for_prospective_use"]:
        detail = "; ".join(result["reasons"][:12]) or "unspecified source-authentication failure"
        raise SourceAuthenticationError(f"prospective source authentication blocked: {detail}")
    return result
