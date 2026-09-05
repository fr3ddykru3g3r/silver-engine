"""Fail-closed AARP-to-HMI identity bridge for the optional AIA track.

This module is intentionally dependency-free.  It does not read an AARP
NetCDF, a FITS cache, a SEP label file, or a locked benchmark partition.  It
only evaluates *already materialized* identity rows and an explicitly supplied
authoritative crosswalk.  In particular, the following are never joins:

* matching a date or a rounded timestamp;
* choosing the nearest observation;
* matching the numeric AARP number to ``HARPNUM``;
* matching row order, filename substrings, or a heuristic region score.

An accepted mapping therefore requires an external authority to prove the
relationship between the canonical AARP source identity, its region number and
observation time, and one exact HMI ``(HARPNUM, T_REC_TAI)`` key.  Missing,
ambiguous, duplicate, or insufficiently proven candidates are rejected and
remain available as an external-only cohort.

The code uses mappings/JSON-shaped records rather than pandas or NumPy so the
contract can be tested on a clean Python installation.  The caller owns any
real data acquisition and must pass only non-test identity metadata during
tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "iris-sep-luna-d-identity-bridge-v1"
RECEIPT_TYPE = "IRIS_SEP_AARP_HMI_IDENTITY_BRIDGE"
MAPPING_METHOD = "authoritative_crosswalk"
EXPECTED_EXACT_KEY_FIELDS = ("aarp_source_id", "harpnum", "t_rec_tai")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AARP_FILENAME = re.compile(
    r"^(?P<date>\d{4}\.\d{2}\.\d{2})_(?P<time>\d{2}:\d{2}:\d{2})_"
    r"7h@1h_(?P<aarp>\d+)\.fits$"
)
_TAI_SUFFIX = re.compile(r"^\S+_TAI$")

# These fields are never allowed in an identity/crosswalk artifact.  This
# prevents a bridge receipt from becoming an accidental tuning input.
_FORBIDDEN_OUTCOME_TOKENS = (
    "future_",
    "_label",
    "label_",
    "outcome",
    "y_true",
    "target",
    "onset_actual",
    "locked_test",
)


class BridgeContractError(ValueError):
    """Raised when an identity bridge candidate violates the contract."""


@dataclass(frozen=True)
class BridgeMapping:
    """A proven one-to-one AARP-to-HMI identity mapping."""

    mapping_id: str
    aarp_source_id: str
    aarp_number: int
    aarp_issue_time_utc: str
    harpnum: int
    t_rec_tai: str
    aarp_provenance: Mapping[str, Any]
    hmi_provenance: Mapping[str, Any]
    authority_provenance: Mapping[str, Any]
    record_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "mapping_id": self.mapping_id,
            "aarp_source_id": self.aarp_source_id,
            "aarp_number": self.aarp_number,
            "aarp_issue_time_utc": self.aarp_issue_time_utc,
            "hmi_key": {
                "harpnum": self.harpnum,
                "t_rec_tai": self.t_rec_tai,
            },
            "provenance": {
                "aarp": dict(self.aarp_provenance),
                "hmi": dict(self.hmi_provenance),
                "authority": dict(self.authority_provenance),
            },
            "record_sha256": self.record_sha256,
            "mapping_method": MAPPING_METHOD,
            "identity_scope": "exact_aarp_source_id_to_exact_hmi_key",
        }


@dataclass(frozen=True)
class BridgeRejection:
    """A machine-readable reason why a source row cannot enter the AIA/HMI bridge."""

    reason_code: str
    detail: str
    aarp_source_id: str | None = None
    hmi_key: Mapping[str, Any] | None = None
    candidate_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "reason_code": self.reason_code,
            "detail": self.detail,
            "candidate_count": self.candidate_count,
        }
        if self.aarp_source_id is not None:
            out["aarp_source_id"] = self.aarp_source_id
        if self.hmi_key is not None:
            out["hmi_key"] = dict(self.hmi_key)
        return out


@dataclass(frozen=True)
class BridgeResult:
    """Result of evaluating one explicitly supplied crosswalk batch."""

    status: str
    fusion_allowed: bool
    accepted: tuple[BridgeMapping, ...]
    rejected: tuple[BridgeRejection, ...]
    input_counts: Mapping[str, int]
    test_outcomes_accessed: bool = False

    def as_dict(self) -> dict[str, Any]:
        if self.accepted and not self.rejected:
            status = "PASS"
        elif self.accepted:
            status = "PARTIAL_PASS"
        else:
            status = "REJECT"
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "fusion_allowed": bool(self.fusion_allowed and self.accepted),
            "fusion_scope": "accepted_mappings_only" if self.accepted else "none",
            "accepted_mappings": [item.as_dict() for item in self.accepted],
            "rejections": [item.as_dict() for item in self.rejected],
            "input_counts": dict(self.input_counts),
            "rules": {
                "accepted_mapping_method": MAPPING_METHOD,
                "exact_hmi_key": list(EXPECTED_EXACT_KEY_FIELDS[1:]),
                "date_only_join": "FORBIDDEN",
                "nearest_timestamp_join": "FORBIDDEN",
                "numeric_aarp_harp_equality": "FORBIDDEN",
                "row_order_or_filename_join": "FORBIDDEN",
                "unmatched_rows": "external_only_not_filled",
            },
            "test_outcomes_accessed": bool(self.test_outcomes_accessed),
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible object deterministically."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BridgeContractError(f"{field} must be a nonempty, trimmed string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    value = _require_nonempty_string(value, field).lower()
    if _SHA256.fullmatch(value) is None:
        raise BridgeContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_utc(value: Any, field: str) -> str:
    text = _require_nonempty_string(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BridgeContractError(f"{field} is not ISO-8601: {text!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BridgeContractError(f"{field} must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    # Source identity uses second precision.  Rejecting fractional seconds
    # avoids silently rounding an exact observation key.
    if parsed.microsecond:
        raise BridgeContractError(f"{field} has fractional seconds")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_tai_key(value: Any, field: str = "t_rec_tai") -> str:
    text = _require_nonempty_string(value, field)
    if _TAI_SUFFIX.fullmatch(text) is None:
        raise BridgeContractError(
            f"{field} must be an exact, whitespace-free TAI key ending in _TAI"
        )
    return text


def _validate_provenance(value: Any, field: str, *, require_artifact_role: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BridgeContractError(f"{field} must be an object")
    required = ("source_uri", "source_version", "source_sha256", "source_row_id", "retrieved_at_utc")
    missing = [name for name in required if name not in value]
    if missing:
        raise BridgeContractError(f"{field} missing provenance fields: {', '.join(missing)}")
    out = dict(value)
    _require_nonempty_string(out["source_uri"], f"{field}.source_uri")
    _require_nonempty_string(out["source_version"], f"{field}.source_version")
    _require_sha256(out["source_sha256"], f"{field}.source_sha256")
    _require_nonempty_string(out["source_row_id"], f"{field}.source_row_id")
    out["retrieved_at_utc"] = _canonical_utc(out["retrieved_at_utc"], f"{field}.retrieved_at_utc")
    if require_artifact_role:
        role = _require_nonempty_string(out.get("artifact_role"), f"{field}.artifact_role")
        if role not in {"identity_crosswalk", "JSOC_metadata_export", "AARP_identity_export"}:
            raise BridgeContractError(f"{field}.artifact_role is not an identity artifact: {role}")
        if "row_sha256" not in out:
            raise BridgeContractError(f"{field}.row_sha256 is required for authority provenance")
        _require_sha256(out["row_sha256"], f"{field}.row_sha256")
    return out


def _assert_metadata_only(value: Any, path: str = "record") -> None:
    """Reject obvious labels/outcomes anywhere in a JSON-shaped record."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            token = str(key).lower()
            if any(forbidden in token for forbidden in _FORBIDDEN_OUTCOME_TOKENS):
                raise BridgeContractError(f"{path}.{key} is outcome/locked-test metadata")
            _assert_metadata_only(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_metadata_only(nested, f"{path}[{index}]")


def _require_integer(value: Any, field: str) -> int:
    # bool is an int subclass but never a valid region identity.
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgeContractError(f"{field} must be an integer")
    if value <= 0:
        raise BridgeContractError(f"{field} must be positive")
    return value


def _parse_canonical_aarp_source_id(value: Any) -> tuple[str, int, str]:
    source_id = _require_nonempty_string(value, "aarp_source_id")
    match = _AARP_FILENAME.fullmatch(source_id)
    if match is None:
        raise BridgeContractError(
            "aarp_source_id must retain the canonical AARP filename identity "
            "YYYY.MM.DD_HH:MM:SS_7h@1h_<number>.fits"
        )
    issue_time = _canonical_utc(
        f"{match.group('date').replace('.', '-')}T{match.group('time')}Z",
        "aarp_source_id.issue_time_utc",
    )
    return source_id, int(match.group("aarp")), issue_time


def validate_aarp_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one AARP source identity without reading feature values."""

    if not isinstance(row, Mapping):
        raise BridgeContractError("AARP identity row must be an object")
    _assert_metadata_only(row, "aarp")
    if str(row.get("partition", "")).strip() == "locked_test":
        raise BridgeContractError("locked_test identity is inaccessible during tuning")
    source_id, encoded_number, encoded_time = _parse_canonical_aarp_source_id(row.get("aarp_source_id"))
    aarp_number = _require_integer(row.get("aarp_number"), "aarp_number")
    if aarp_number != encoded_number:
        raise BridgeContractError("aarp_number disagrees with canonical aarp_source_id")
    issue_time = _canonical_utc(row.get("issue_time_utc"), "issue_time_utc")
    if issue_time != encoded_time:
        raise BridgeContractError("issue_time_utc disagrees with canonical AARP source identity")
    provenance = _validate_provenance(row.get("provenance"), "aarp.provenance")
    return {
        "aarp_source_id": source_id,
        "aarp_number": aarp_number,
        "issue_time_utc": issue_time,
        "provenance": provenance,
    }


def validate_hmi_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one exact HMI identity key and its source provenance."""

    if not isinstance(row, Mapping):
        raise BridgeContractError("HMI identity row must be an object")
    _assert_metadata_only(row, "hmi")
    if str(row.get("partition", "")).strip() == "locked_test":
        raise BridgeContractError("locked_test identity is inaccessible during tuning")
    harpnum = _require_integer(row.get("harpnum"), "harpnum")
    t_rec_tai = _validate_tai_key(row.get("t_rec_tai"))
    hmi_record_id = _require_nonempty_string(row.get("hmi_record_id"), "hmi_record_id")
    provenance = _validate_provenance(row.get("provenance"), "hmi.provenance")
    return {
        "harpnum": harpnum,
        "t_rec_tai": t_rec_tai,
        "hmi_record_id": hmi_record_id,
        "provenance": provenance,
    }


def _validate_authority_evidence(
    candidate: Mapping[str, Any], aarp: Mapping[str, Any], hmi: Mapping[str, Any]
) -> dict[str, Any]:
    """Require source-backed proof of region and observation-time identity."""

    method = candidate.get("mapping_method")
    if method != MAPPING_METHOD:
        raise BridgeContractError(
            "only mapping_method='authoritative_crosswalk' is admissible; "
            "date, nearest-time, filename, row-order, and numeric-equality joins are rejected"
        )
    exact_fields = candidate.get("exact_key_fields")
    if tuple(exact_fields or ()) != EXPECTED_EXACT_KEY_FIELDS:
        raise BridgeContractError(
            "exact_key_fields must be [aarp_source_id, harpnum, t_rec_tai] in that order"
        )
    if candidate.get("ambiguity_status") != "unambiguous":
        raise BridgeContractError("crosswalk candidate is not declared unambiguous")
    if candidate.get("candidate_count") != 1:
        raise BridgeContractError("crosswalk candidate_count must be exactly 1")

    source_id = candidate.get("aarp_source_id")
    if source_id != aarp["aarp_source_id"]:
        raise BridgeContractError("crosswalk AARP source identity does not match cohort row")
    if candidate.get("aarp_number") != aarp["aarp_number"]:
        raise BridgeContractError("crosswalk AARP region number does not match source identity")
    if candidate.get("harpnum") != hmi["harpnum"]:
        raise BridgeContractError("crosswalk HARPNUM does not match HMI identity row")
    if candidate.get("t_rec_tai") != hmi["t_rec_tai"]:
        raise BridgeContractError("crosswalk T_REC_TAI is not the exact HMI key")

    authority = _validate_provenance(
        candidate.get("authority_provenance"),
        "crosswalk.authority_provenance",
        require_artifact_role=True,
    )
    time_proof = candidate.get("time_proof")
    if not isinstance(time_proof, Mapping):
        raise BridgeContractError("time_proof is required")
    if time_proof.get("kind") != "authoritative_exact_observation":
        raise BridgeContractError("time_proof must be authoritative_exact_observation")
    if _canonical_utc(time_proof.get("aarp_issue_time_utc"), "time_proof.aarp_issue_time_utc") != aarp[
        "issue_time_utc"
    ]:
        raise BridgeContractError("time proof does not match AARP issue time")
    if time_proof.get("hmi_t_rec_tai") != hmi["t_rec_tai"]:
        raise BridgeContractError("time proof does not match exact HMI T_REC_TAI")
    if time_proof.get("authority_row_id") != authority["source_row_id"]:
        raise BridgeContractError("time proof authority row differs from crosswalk provenance")

    region_proof = candidate.get("region_proof")
    if not isinstance(region_proof, Mapping):
        raise BridgeContractError("region_proof is required")
    if region_proof.get("kind") != "authoritative_region_crosswalk":
        raise BridgeContractError("region_proof must be authoritative_region_crosswalk")
    if region_proof.get("aarp_number") != aarp["aarp_number"]:
        raise BridgeContractError("region proof does not match AARP region number")
    if region_proof.get("harpnum") != hmi["harpnum"]:
        raise BridgeContractError("region proof does not match HMI HARPNUM")
    if region_proof.get("authority_row_id") != authority["source_row_id"]:
        raise BridgeContractError("region proof authority row differs from crosswalk provenance")
    return authority


def _reject_from_error(
    exc: Exception,
    *,
    source_id: str | None = None,
    hmi_key: Mapping[str, Any] | None = None,
    candidate_count: int = 0,
) -> BridgeRejection:
    detail = str(exc)
    lower = detail.lower()
    if "authoritative_crosswalk" in lower or "mapping_method" in lower:
        reason = "NON_AUTHORITATIVE_MAPPING_METHOD"
    elif "ambiguous" in lower or "candidate_count" in lower:
        reason = "AMBIGUOUS_CROSSWALK"
    elif "exact hmi" in lower or "t_rec_tai" in lower or "harpnum" in lower:
        reason = "EXACT_HMI_KEY_INVALID"
    elif "provenance" in lower or "proof" in lower:
        reason = "INSUFFICIENT_PROVENANCE"
    else:
        reason = "IDENTITY_CONTRACT_INVALID"
    return BridgeRejection(reason, detail, source_id, hmi_key, candidate_count)


def evaluate_bridge(
    aarp_rows: Iterable[Mapping[str, Any]],
    hmi_rows: Iterable[Mapping[str, Any]],
    mapping_rows: Iterable[Mapping[str, Any]],
    *,
    phase: str = "tuning",
) -> BridgeResult:
    """Evaluate an explicit AARP/HMI crosswalk and fail closed on every defect.

    Parameters are metadata rows only.  ``phase`` is intentionally tuning-only
    in this workstream; the primary agent owns any eventual final locked-test
    evaluation.  Every accepted mapping is one-to-one and retains all three
    provenance records.  AARP rows without a valid mapping are rejected rather
    than filled by a neighboring observation.
    """

    if phase != "tuning":
        raise BridgeContractError("Luna D bridge is tuning-only; locked evaluation is primary-agent work")

    aarp_input = list(aarp_rows)
    hmi_input = list(hmi_rows)
    mapping_input = list(mapping_rows)
    rejected: list[BridgeRejection] = []

    aarp_index: dict[str, dict[str, Any]] = {}
    invalid_aarp_sources: set[str] = set()
    for index, row in enumerate(aarp_input):
        try:
            parsed = validate_aarp_identity(row)
        except BridgeContractError as exc:
            rejected.append(_reject_from_error(exc))
            continue
        source_id = parsed["aarp_source_id"]
        if source_id in aarp_index:
            rejected.append(
                BridgeRejection(
                    "DUPLICATE_AARP_SOURCE_ID",
                    f"duplicate canonical AARP source identity: {source_id}",
                    source_id=source_id,
                )
            )
            # A duplicate source identity invalidates *both* copies.  Keeping
            # the first copy here would let incidental row order establish an
            # identity and could produce a false one-to-one bridge.
            aarp_index.pop(source_id, None)
            invalid_aarp_sources.add(source_id)
            continue
        aarp_index[source_id] = parsed

    hmi_index: dict[tuple[int, str], dict[str, Any]] = {}
    for row in hmi_input:
        try:
            parsed = validate_hmi_identity(row)
        except BridgeContractError as exc:
            rejected.append(_reject_from_error(exc))
            continue
        key = (parsed["harpnum"], parsed["t_rec_tai"])
        if key in hmi_index:
            rejected.append(
                BridgeRejection(
                    "DUPLICATE_HMI_KEY",
                    f"duplicate exact HMI key: {key[0]} / {key[1]}",
                    hmi_key={"harpnum": key[0], "t_rec_tai": key[1]},
                )
            )
            # Remove the key so no candidate can accidentally use a duplicate.
            hmi_index.pop(key, None)
            continue
        hmi_index[key] = parsed

    candidates_by_aarp: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in mapping_input:
        if not isinstance(candidate, Mapping):
            rejected.append(BridgeRejection("CROSSWALK_ROW_INVALID", "crosswalk row is not an object"))
            continue
        try:
            _assert_metadata_only(candidate, "crosswalk")
        except BridgeContractError as exc:
            rejected.append(_reject_from_error(exc))
            continue
        source_id = candidate.get("aarp_source_id")
        if not isinstance(source_id, str):
            rejected.append(BridgeRejection("CROSSWALK_ROW_INVALID", "crosswalk has no AARP source identity"))
            continue
        candidates_by_aarp.setdefault(source_id, []).append(candidate)

    accepted: list[BridgeMapping] = []
    used_hmi: dict[tuple[int, str], str] = {}
    # Iterate in canonical source order, never in the source row's incidental
    # order.  Ordering does not establish identity; it only stabilizes receipts.
    for source_id in sorted(aarp_index):
        aarp = aarp_index[source_id]
        candidates = candidates_by_aarp.get(source_id, [])
        if not candidates:
            rejected.append(
                BridgeRejection(
                    "NO_AUTHORITATIVE_CROSSWALK",
                    "no explicit authoritative crosswalk candidate was supplied; row remains external-only",
                    aarp_source_id=source_id,
                )
            )
            continue
        if len(candidates) != 1:
            rejected.append(
                BridgeRejection(
                    "AMBIGUOUS_CROSSWALK",
                    "more than one crosswalk candidate exists for the canonical AARP source identity",
                    aarp_source_id=source_id,
                    candidate_count=len(candidates),
                )
            )
            continue
        candidate = candidates[0]
        hmi_key = None
        if isinstance(candidate.get("harpnum"), int) and isinstance(candidate.get("t_rec_tai"), str):
            hmi_key = {"harpnum": candidate["harpnum"], "t_rec_tai": candidate["t_rec_tai"]}
        try:
            if candidate.get("harpnum") is not None and candidate.get("t_rec_tai") is not None:
                hmi_lookup_key = (candidate["harpnum"], candidate["t_rec_tai"])
                hmi = hmi_index.get(hmi_lookup_key)
            else:
                hmi_lookup_key = None
                hmi = None
            if hmi is None:
                raise BridgeContractError(
                    "crosswalk references an HMI key that is absent from the supplied HMI identity manifest"
                )
            authority = _validate_authority_evidence(candidate, aarp, hmi)
            assert hmi_lookup_key is not None
            if hmi_lookup_key in used_hmi:
                raise BridgeContractError(
                    "exact HMI key is already assigned to another canonical AARP source identity"
                )
            mapping_id = _require_nonempty_string(candidate.get("mapping_id"), "mapping_id")
            canonical_mapping = {
                "mapping_id": mapping_id,
                "aarp_source_id": aarp["aarp_source_id"],
                "aarp_number": aarp["aarp_number"],
                "aarp_issue_time_utc": aarp["issue_time_utc"],
                "harpnum": hmi["harpnum"],
                "t_rec_tai": hmi["t_rec_tai"],
                "aarp_provenance": aarp["provenance"],
                "hmi_provenance": hmi["provenance"],
                "authority_provenance": authority,
                "mapping_method": MAPPING_METHOD,
            }
            accepted.append(
                BridgeMapping(
                    mapping_id=mapping_id,
                    aarp_source_id=aarp["aarp_source_id"],
                    aarp_number=aarp["aarp_number"],
                    aarp_issue_time_utc=aarp["issue_time_utc"],
                    harpnum=hmi["harpnum"],
                    t_rec_tai=hmi["t_rec_tai"],
                    aarp_provenance=aarp["provenance"],
                    hmi_provenance=hmi["provenance"],
                    authority_provenance=authority,
                    record_sha256=sha256_json(canonical_mapping),
                )
            )
            used_hmi[hmi_lookup_key] = source_id
        except BridgeContractError as exc:
            rejection = _reject_from_error(
                exc,
                source_id=source_id,
                hmi_key=hmi_key,
                candidate_count=1,
            )
            # A reused HMI key gets a distinct, stable reason code.
            if "already assigned" in str(exc):
                rejection = BridgeRejection(
                    "HMI_KEY_REUSED",
                    str(exc),
                    aarp_source_id=source_id,
                    hmi_key=hmi_key,
                    candidate_count=1,
                )
            elif "absent from" in str(exc):
                rejection = BridgeRejection(
                    "HMI_KEY_NOT_IN_MANIFEST",
                    str(exc),
                    aarp_source_id=source_id,
                    hmi_key=hmi_key,
                    candidate_count=1,
                )
            rejected.append(rejection)

    # Crosswalk rows that refer to a source identity outside the supplied
    # cohort are rejected.  They are never allowed to enlarge a frozen cohort.
    for source_id, candidates in sorted(candidates_by_aarp.items()):
        if source_id not in aarp_index:
            reason = (
                "DUPLICATE_AARP_SOURCE_ID"
                if source_id in invalid_aarp_sources
                else "AARP_SOURCE_NOT_IN_MANIFEST"
            )
            rejected.append(
                BridgeRejection(
                    reason,
                    (
                        "canonical AARP source identity is duplicated in the supplied manifest"
                        if source_id in invalid_aarp_sources
                        else "crosswalk source identity is absent from the supplied AARP manifest"
                    ),
                    aarp_source_id=source_id,
                    candidate_count=len(candidates),
                )
            )

    accepted_ids = {item.mapping_id for item in accepted}
    if len(accepted_ids) != len(accepted):
        # This should be impossible for valid input, but fail closed if future
        # callers alter the implementation.
        raise BridgeContractError("accepted mapping_id values are not unique")

    return BridgeResult(
        status="PASS" if accepted and not rejected else "PARTIAL_PASS" if accepted else "REJECT",
        fusion_allowed=bool(accepted),
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        input_counts={
            "aarp_identity_rows": len(aarp_input),
            "hmi_identity_rows": len(hmi_input),
            "crosswalk_rows": len(mapping_input),
            "accepted_mappings": len(accepted),
            "rejected_decisions": len(rejected),
        },
        test_outcomes_accessed=False,
    )


def build_bridge_receipt(
    result: BridgeResult | Mapping[str, Any],
    *,
    generated_at_utc: str,
    parent_contract_id: str = "iris-sep-sepval-v1",
    aarp_audit_reference: str = "../../../../source_snapshot/iris-model/flare_system/aarp.py",
    architecture_reference: str = "../../../architecture/ADR-002-aarp-aia-feature-expert.md",
) -> dict[str, Any]:
    """Build a JSON receipt suitable for an immutable workstream artifact."""

    if isinstance(result, BridgeResult):
        body = result.as_dict()
    elif isinstance(result, Mapping):
        body = dict(result)
    else:
        raise BridgeContractError("result must be BridgeResult or a JSON-shaped mapping")
    generated = _canonical_utc(generated_at_utc, "generated_at_utc")
    receipt = {
        "receipt_type": RECEIPT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "workstream": "luna_d",
        "generated_at_utc": generated,
        "parent_contract_id": parent_contract_id,
        "status": body.get("status", "REJECT"),
        "fusion_allowed": bool(body.get("fusion_allowed", False)),
        "decision": body,
        "provenance_basis": {
            "aarp_audit_reference": aarp_audit_reference,
            "architecture_reference": architecture_reference,
            "accepted_mapping_provenance_required": True,
            "authority_row_and_artifact_hash_required": True,
        },
        "scientific_boundaries": {
            "date_only_join": "REJECTED",
            "nearest_timestamp_join": "REJECTED",
            "numeric_aarp_harp_equality": "REJECTED",
            "unmatched_aia_examples": "external_only",
            "aia_pretraining_sep_labels": "forbidden",
            "locked_test_outcomes_accessed": False,
        },
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def build_no_crosswalk_result() -> BridgeResult:
    """Return the explicit current-state rejection used before a crosswalk exists."""

    return BridgeResult(
        status="REJECT",
        fusion_allowed=False,
        accepted=(),
        rejected=(
            BridgeRejection(
                "NO_AUTHORITATIVE_CROSSWALK",
                "No authoritative AARP-to-HMI crosswalk has been supplied; AIA/HMI fusion is disabled.",
            ),
        ),
        input_counts={
            "aarp_identity_rows": 0,
            "hmi_identity_rows": 0,
            "crosswalk_rows": 0,
            "accepted_mappings": 0,
            "rejected_decisions": 1,
        },
        test_outcomes_accessed=False,
    )


__all__ = [
    "BridgeContractError",
    "BridgeMapping",
    "BridgeRejection",
    "BridgeResult",
    "EXPECTED_EXACT_KEY_FIELDS",
    "MAPPING_METHOD",
    "SCHEMA_VERSION",
    "build_bridge_receipt",
    "build_no_crosswalk_result",
    "evaluate_bridge",
    "sha256_json",
    "validate_aarp_identity",
    "validate_hmi_identity",
]
