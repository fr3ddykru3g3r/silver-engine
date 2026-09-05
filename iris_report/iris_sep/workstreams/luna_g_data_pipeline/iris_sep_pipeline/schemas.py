"""Strict records and forecast-time availability validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re
from typing import Iterable, Mapping, Optional

from .errors import DuplicateRecordError, PipelineError, ProtectedDataError

UTC = timezone.utc
_SOURCE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PipelineError("timestamps must be timezone-aware UTC datetimes")
    result = value.astimezone(UTC)
    if result.utcoffset() != UTC.utcoffset(result):
        raise PipelineError("timestamp conversion failed")
    return result


def iso_z(value: datetime) -> str:
    value = utc(value)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_issue_id(source: str, issue_time: datetime) -> str:
    if not isinstance(source, str) or not _SOURCE_RE.fullmatch(source):
        raise PipelineError("source must be lowercase and contain only [a-z0-9_.-]")
    return f"{source}@{iso_z(issue_time)}"


@dataclass(frozen=True)
class IssueRecord:
    issue_id: str
    source: str
    issue_time: datetime
    source_revision: str = "synthetic-v1"
    origin: str = "synthetic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue_time", utc(self.issue_time))
        expected = canonical_issue_id(self.source, self.issue_time)
        if self.issue_id != expected:
            raise PipelineError(f"non-canonical issue_id: expected {expected}, got {self.issue_id}")
        if self.origin != "synthetic":
            raise ProtectedDataError("only synthetic records are accepted by this workstream")


def make_issue(source: str, issue_time: datetime, **kwargs: str) -> IssueRecord:
    return IssueRecord(canonical_issue_id(source, issue_time), source, issue_time, **kwargs)


def validate_issues(records: Iterable[IssueRecord]) -> list[IssueRecord]:
    result = list(records)
    ids: set[str] = set()
    keys: set[tuple[str, datetime]] = set()
    for row in result:
        if not isinstance(row, IssueRecord):
            raise PipelineError("issues must be IssueRecord instances")
        key = (row.source, utc(row.issue_time))
        if row.issue_id in ids or key in keys:
            raise DuplicateRecordError(f"duplicate issue identity: {row.issue_id}")
        ids.add(row.issue_id)
        keys.add(key)
    return sorted(result, key=lambda r: (r.issue_time, r.issue_id))


@dataclass(frozen=True)
class FeatureRecord:
    issue_id: str
    feature_name: str
    source_time: datetime
    publication_time: datetime
    value: Optional[float]
    max_latency_hours: float = 6.0
    modality: str = "synthetic"
    origin: str = "synthetic"

    def __post_init__(self) -> None:
        source = utc(self.source_time)
        publication = utc(self.publication_time)
        object.__setattr__(self, "source_time", source)
        object.__setattr__(self, "publication_time", publication)
        if not self.feature_name or not isinstance(self.feature_name, str):
            raise PipelineError("feature_name is required")
        if not math.isfinite(self.max_latency_hours) or self.max_latency_hours < 0:
            raise PipelineError("max_latency_hours must be finite and non-negative")
        if self.value is not None and (not isinstance(self.value, (int, float)) or not math.isfinite(float(self.value))):
            raise PipelineError("feature values must be finite numbers or None")
        if self.origin != "synthetic":
            raise ProtectedDataError("only synthetic feature records are accepted")


@dataclass(frozen=True)
class Observation:
    observed_time: datetime
    flux_pfu: float
    origin: str = "synthetic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_time", utc(self.observed_time))
        if not math.isfinite(float(self.flux_pfu)) or self.flux_pfu < 0:
            raise PipelineError("particle flux must be a finite non-negative number")
        if self.origin != "synthetic":
            raise ProtectedDataError("only synthetic observations are accepted")


def validate_features(issues: Iterable[IssueRecord], features: Iterable[FeatureRecord]) -> list[FeatureRecord]:
    issue_rows = validate_issues(issues)
    issue_map = {row.issue_id: row for row in issue_rows}
    result = list(features)
    seen: set[tuple[str, str]] = set()
    for row in result:
        if not isinstance(row, FeatureRecord):
            raise PipelineError("features must be FeatureRecord instances")
        if row.issue_id not in issue_map:
            raise PipelineError(f"feature references unknown issue: {row.issue_id}")
        identity = (row.issue_id, row.feature_name)
        if identity in seen:
            raise DuplicateRecordError(f"duplicate feature identity: {identity}")
        seen.add(identity)
        issue_time = issue_map[row.issue_id].issue_time
        if row.source_time > row.publication_time:
            raise PipelineError("publication_time cannot precede source_time")
        if row.publication_time > issue_time:
            raise PipelineError("feature was published after forecast issue time")
        latency_hours = (row.publication_time - row.source_time).total_seconds() / 3600.0
        if latency_hours > row.max_latency_hours + 1e-9:
            raise PipelineError("publication latency exceeds the frozen feature limit")
    return sorted(result, key=lambda r: (r.issue_id, r.feature_name))


@dataclass(frozen=True)
class TargetRecord:
    issue_id: str
    label: Optional[int]
    crossing_time: Optional[datetime] = None
    excluded_reason: Optional[str] = None
    episode_id: Optional[str] = None


@dataclass(frozen=True)
class CohortUnit:
    unit_id: str
    kind: str
    issue_ids: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    label: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start_time", utc(self.start_time))
        object.__setattr__(self, "end_time", utc(self.end_time))
        if self.kind not in {"episode", "quiet_block"}:
            raise PipelineError("cohort unit kind must be episode or quiet_block")
        if self.end_time < self.start_time or not self.issue_ids:
            raise PipelineError("cohort unit interval or issue_ids is invalid")
