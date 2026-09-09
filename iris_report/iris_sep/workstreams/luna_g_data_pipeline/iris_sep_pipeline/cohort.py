"""Target construction, event/quiet grouping, and chronological partitions."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Iterable, Mapping

from .errors import DuplicateRecordError, PipelineError
from .schemas import CohortUnit, IssueRecord, Observation, TargetRecord, utc, validate_issues

UTC = timezone.utc


def _ordered_observations(observations: Iterable[Observation]) -> list[Observation]:
    result = sorted(observations, key=lambda row: row.observed_time)
    seen: set[datetime] = set()
    for row in result:
        if row.observed_time in seen:
            raise DuplicateRecordError(f"duplicate observation timestamp: {row.observed_time!r}")
        seen.add(row.observed_time)
    return result


def _first_crossing(observations: list[Observation], issue_time: datetime, threshold: float, horizon: timedelta) -> datetime | None:
    prior = [row for row in observations if row.observed_time <= issue_time]
    if not prior:
        return None
    previous = prior[-1]
    if previous.flux_pfu >= threshold:
        return None
    for current in observations:
        if current.observed_time <= issue_time:
            continue
        if current.observed_time > issue_time + horizon:
            break
        if previous.flux_pfu < threshold <= current.flux_pfu:
            return current.observed_time
        previous = current
    return None


def build_targets(
    issues: Iterable[IssueRecord],
    observations: Iterable[Observation],
    *,
    threshold_pfu: float = 10.0,
    horizon_hours: float = 24.0,
) -> list[TargetRecord]:
    """Create only at-risk, new-crossing labels.

    An issue is excluded when the latest available particle observation at the
    issue time is already above threshold, or when there is no pre-issue
    observation to establish the at-risk state.  This avoids relabelling an
    ongoing storm as a new prediction opportunity.
    """
    if threshold_pfu <= 0 or horizon_hours <= 0:
        raise PipelineError("threshold and horizon must be positive")
    issue_rows = validate_issues(issues)
    obs = _ordered_observations(observations)
    horizon = timedelta(hours=horizon_hours)
    result: list[TargetRecord] = []
    for issue in issue_rows:
        prior = [row for row in obs if row.observed_time <= issue.issue_time]
        if not prior:
            result.append(TargetRecord(issue.issue_id, None, excluded_reason="no_preissue_observation"))
            continue
        if prior[-1].flux_pfu >= threshold_pfu:
            result.append(TargetRecord(issue.issue_id, None, excluded_reason="already_above_threshold"))
            continue
        crossing = _first_crossing(obs, issue.issue_time, threshold_pfu, horizon)
        result.append(TargetRecord(issue.issue_id, int(crossing is not None), crossing_time=crossing))
    return result


def _episode_runs(observations: list[Observation], threshold_pfu: float, max_gap_hours: float) -> list[tuple[str, datetime, datetime]]:
    """Return deterministic above-threshold runs as (id, start, end)."""
    runs: list[list[Observation]] = []
    current_run: list[Observation] = []
    max_gap = timedelta(hours=max_gap_hours)
    previous_observation: Observation | None = None
    for row in observations:
        above = row.flux_pfu >= threshold_pfu
        contiguous = (
            previous_observation is not None
            and row.observed_time - previous_observation.observed_time <= max_gap
        )
        if above and current_run and contiguous:
            current_run.append(row)
        elif above:
            if current_run:
                runs.append(current_run)
            current_run = [row]
        else:
            if current_run:
                runs.append(current_run)
                current_run = []
        previous_observation = row
    if current_run:
        runs.append(current_run)
    result = []
    for run in runs:
        start, end = run[0].observed_time, run[-1].observed_time
        digest = hashlib.sha256(start.isoformat().encode("ascii")).hexdigest()[:12]
        result.append((f"episode-{digest}", start, end))
    return result


def build_cohort_units(
    issues: Iterable[IssueRecord],
    targets: Iterable[TargetRecord],
    observations: Iterable[Observation],
    *,
    threshold_pfu: float = 10.0,
    cadence_hours: float = 1.0,
    horizon_hours: float = 24.0,
) -> list[CohortUnit]:
    """Group positive predictions by complete episode and negatives by quiet block.

    Excluded rows are not cohort units.  A quiet block is a maximal contiguous
    run of eligible negative issue times, split around any forecast window that
    reaches an observed episode.  Units are the atomic partitioning objects.
    """
    issue_rows = validate_issues(issues)
    issue_map = {row.issue_id: row for row in issue_rows}
    target_map: dict[str, TargetRecord] = {}
    for target in targets:
        if target.issue_id in target_map:
            raise DuplicateRecordError(f"duplicate target: {target.issue_id}")
        if target.issue_id not in issue_map:
            raise PipelineError(f"target references unknown issue: {target.issue_id}")
        target_map[target.issue_id] = target
    if set(target_map) != set(issue_map):
        raise PipelineError("every issue must have exactly one target record")
    obs = _ordered_observations(observations)
    episodes = _episode_runs(obs, threshold_pfu, max(2.0, cadence_hours * 1.5))
    positive: dict[str, list[str]] = defaultdict(list)
    negative: list[IssueRecord] = []
    for issue in issue_rows:
        target = target_map[issue.issue_id]
        if target.label is None:
            continue
        if target.label == 1:
            if target.crossing_time is None:
                raise PipelineError("positive target is missing crossing_time")
            candidates = [ep for ep in episodes if ep[1] <= target.crossing_time <= ep[2]]
            if not candidates:
                raise PipelineError("positive crossing does not map to an observed episode")
            positive[candidates[0][0]].append(issue.issue_id)
        elif target.label == 0:
            negative.append(issue)
        else:
            raise PipelineError("target labels must be 0, 1, or None")
    units: list[CohortUnit] = []
    for episode_id, ids in positive.items():
        rows = [issue_map[i] for i in ids]
        ep = next(ep for ep in episodes if ep[0] == episode_id)
        units.append(CohortUnit(episode_id, "episode", tuple(sorted(ids)), min(r.issue_time for r in rows), max(r.issue_time for r in rows), 1))

    # Quiet blocks cannot straddle an event's forecast window.  A one-hour
    # cadence is required by the frozen contract; larger gaps split blocks.
    max_gap = timedelta(hours=cadence_hours * 1.5)
    negative.sort(key=lambda row: row.issue_time)
    block: list[IssueRecord] = []
    block_no = 0
    def flush() -> None:
        nonlocal block, block_no
        if not block:
            return
        block_no += 1
        start, end = block[0].issue_time, block[-1].issue_time
        digest = hashlib.sha256(start.isoformat().encode("ascii")).hexdigest()[:12]
        units.append(CohortUnit(f"quiet-{digest}-{block_no:03d}", "quiet_block", tuple(r.issue_id for r in block), start, end, 0))
        block = []

    for issue in negative:
        crosses_episode_window = any(
            issue.issue_time < ep_end + timedelta(hours=horizon_hours)
            and issue.issue_time + timedelta(hours=horizon_hours) >= ep_start
            for _, ep_start, ep_end in episodes
        )
        if block and (issue.issue_time - block[-1].issue_time > max_gap or crosses_episode_window):
            flush()
        if not crosses_episode_window:
            block.append(issue)
    flush()
    return sorted(units, key=lambda unit: (unit.start_time, unit.unit_id))


ROLE_ORDER = ("train", "validation_monitor", "validation_calibration", "validation_threshold", "locked_test")


def assign_chronological_roles(
    units: Iterable[CohortUnit],
    role_counts: Mapping[str, int],
    *,
    purge_hours: float = 24.0,
) -> dict[str, tuple[CohortUnit, ...]]:
    """Assign whole units chronologically, dropping a 24h purge at each boundary.

    ``role_counts`` describes provisional chronological capacity, not a promise
    that every requested row survives purging.  Purged units are not assigned
    to any role and are reported under ``purged``.  The returned mapping is
    fail-closed: every non-empty adjacent role has at least the requested gap.
    """
    if purge_hours < 0:
        raise PipelineError("purge_hours must be non-negative")
    unknown = set(role_counts) - set(ROLE_ORDER)
    if unknown or any(role_counts.get(role, 0) < 0 for role in role_counts):
        raise PipelineError("invalid role_counts")
    ordered = sorted(units, key=lambda unit: (unit.start_time, unit.unit_id))
    if len({unit.unit_id for unit in ordered}) != len(ordered):
        raise DuplicateRecordError("duplicate cohort unit id")
    if sum(role_counts.get(role, 0) for role in ROLE_ORDER) > len(ordered):
        raise PipelineError("role_counts exceed available cohort units")
    slices: dict[str, list[CohortUnit]] = {}
    cursor = 0
    for role in ROLE_ORDER:
        count = role_counts.get(role, 0)
        slices[role] = ordered[cursor:cursor + count]
        cursor += count
    purged: set[str] = set()
    purge = timedelta(hours=purge_hours)
    result: dict[str, tuple[CohortUnit, ...]] = {}
    previous_role_end: datetime | None = None
    for role in ROLE_ORDER:
        kept: list[CohortUnit] = []
        for unit in slices[role]:
            # Forecast targets include a crossing exactly at issue+horizon.
            # Purging is therefore strict: an adjacent role may not begin at
            # the preceding role's inclusive horizon endpoint.
            if previous_role_end is not None and unit.start_time <= previous_role_end + purge:
                purged.add(unit.unit_id)
            else:
                kept.append(unit)
        result[role] = tuple(kept)
        if kept:
            previous_role_end = max(unit.end_time for unit in kept)
    # Keep the audit trail in the returned object without inventing a sixth
    # model role; callers can serialize this reserved key in a partition manifest.
    result["purged"] = tuple(unit for unit in ordered if unit.unit_id in purged)
    active_roles = [role for role in ROLE_ORDER if result[role]]
    for left, right in zip(active_roles, active_roles[1:]):
        if max(u.end_time for u in result[left]) + purge >= min(u.start_time for u in result[right]):
            raise PipelineError(f"purge boundary violated between {left} and {right}")
    return result
