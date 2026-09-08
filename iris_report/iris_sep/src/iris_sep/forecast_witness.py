"""External server-witness receipts for prospective IRIS-SEP forecasts.

A Git commit timestamp or caller-provided ``sealed_at`` field is not independent
evidence that a forecast digest existed before its outcome. This module binds an
already-created forecast/comparison seal to the exact body of a GitHub issue/PR
comment and validates the server-assigned ``created_at`` *and* ``updated_at``
timestamps returned by GitHub.

A usable witness must be immutable after creation: ``updated_at`` must equal
``created_at``. This deliberately rejects comments edited after creation even if
the final body exactly matches the requested witness text. Otherwise a harmless
comment could be created before the outcome and rewritten after the outcome to
claim an earlier forecast.

The witness proves only pre-outcome existence of the bound digest. It does not
attest that source bytes came from a provider, that model inputs were causal, or
that evaluation was custodian-blinded.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from typing import Any, Mapping
from urllib.parse import urlparse


FORMAT = "IRIS_SEP_GITHUB_FORECAST_WITNESS_V2"
MAX_WITNESS_DELAY = timedelta(minutes=5)


class ForecastWitnessError(ValueError):
    """Raised when a server witness cannot establish timely immutable publication."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForecastWitnessError("witness payload is not canonical JSON") from exc


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ForecastWitnessError(f"{name} must be lowercase SHA-256")
    return value


def _time(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ForecastWitnessError(f"{name} is invalid") from exc
    else:
        raise ForecastWitnessError(f"{name} must be datetime/ISO-8601")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ForecastWitnessError(f"{name} must be timezone-aware")
    return parsed


def build_witness_statement(
    *,
    issued_at: datetime,
    forecast_seal_sha256: str,
    package_manifest_sha256: str,
    comparison_seal_sha256: str | None = None,
) -> dict[str, Any]:
    issue = _time(issued_at, "issued_at")
    payload = {
        "purpose": "IRIS_SEP_PRE_OUTCOME_FORECAST_EXISTENCE_WITNESS",
        "issued_at": issue.isoformat(),
        "forecast_seal_sha256": _sha(forecast_seal_sha256, "forecast_seal_sha256"),
        "package_manifest_sha256": _sha(package_manifest_sha256, "package_manifest_sha256"),
        "comparison_seal_sha256": (
            None if comparison_seal_sha256 is None else _sha(comparison_seal_sha256, "comparison_seal_sha256")
        ),
        "claim_boundary": (
            "Server timestamp witnesses digest existence only; it does not attest provider bytes, "
            "causal input truth, model correctness, independence, or forecast skill."
        ),
    }
    payload["statement_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def witness_comment_body(statement: Mapping[str, Any]) -> str:
    if not isinstance(statement, Mapping):
        raise ForecastWitnessError("statement must be a mapping")
    unsigned = dict(statement)
    claimed = unsigned.pop("statement_sha256", None)
    if _sha(claimed, "statement_sha256") != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise ForecastWitnessError("statement digest mismatch")
    return "IRIS-SEP PRE-OUTCOME WITNESS\n" + _canonical_json(dict(statement)).decode("utf-8")


def validate_github_comment_witness(
    *,
    statement: Mapping[str, Any],
    github_comment: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact body plus immutable GitHub server timestamps.

    ``updated_at == created_at`` is required. GitHub's issue-comment resource
    exposes both values server-side, so an edited comment is not an admissible
    pre-outcome witness even when its final body is correct.
    """
    expected_body = witness_comment_body(statement)
    if not isinstance(github_comment, Mapping):
        raise ForecastWitnessError("github_comment must be a mapping")
    if github_comment.get("body") != expected_body:
        raise ForecastWitnessError("GitHub witness body does not exactly match statement")
    comment_id = github_comment.get("id")
    if not isinstance(comment_id, int) or comment_id <= 0:
        raise ForecastWitnessError("GitHub witness comment id is invalid")

    created = _time(github_comment.get("created_at"), "GitHub created_at")
    updated = _time(github_comment.get("updated_at"), "GitHub updated_at")
    if updated != created:
        raise ForecastWitnessError("GitHub witness comment was edited after creation")

    issue = _time(statement.get("issued_at"), "statement issued_at")
    if created < issue:
        raise ForecastWitnessError("GitHub witness predates forecast issue")
    if created - issue > MAX_WITNESS_DELAY:
        raise ForecastWitnessError("GitHub witness was published more than five minutes after issue")

    html_url = github_comment.get("html_url")
    if not isinstance(html_url, str):
        raise ForecastWitnessError("GitHub witness html_url missing")
    parsed = urlparse(html_url)
    if parsed.scheme != "https" or parsed.hostname != "github.com":
        raise ForecastWitnessError("GitHub witness URL is not an HTTPS github.com record")

    receipt = {
        "format": FORMAT,
        "statement": dict(statement),
        "statement_sha256": str(statement["statement_sha256"]),
        "github_comment_id": comment_id,
        "github_comment_created_at": created.isoformat(),
        "github_comment_updated_at": updated.isoformat(),
        "github_comment_unedited_since_creation": True,
        "github_comment_html_url": html_url,
        "github_comment_body_sha256": hashlib.sha256(expected_body.encode("utf-8")).hexdigest(),
        "witness_delay_seconds": float((created - issue).total_seconds()),
        "server_timestamp_provider": "GitHub issue-comment created_at/updated_at",
        "pre_outcome_digest_existence_witnessed": True,
        "provider_source_attestation_verified": False,
        "custodian_blinding_verified": False,
        "independence_verified": False,
    }
    receipt["witness_receipt_sha256"] = hashlib.sha256(_canonical_json(receipt)).hexdigest()
    return receipt
