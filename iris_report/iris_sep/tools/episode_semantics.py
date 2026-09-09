from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OperationalEpisode:
    episode_id: str
    start: pd.Timestamp
    end: pd.Timestamp

    def validate(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("episode timestamps must be timezone aware")
        if self.end <= self.start:
            raise ValueError("episode end must be after start")


@dataclass(frozen=True)
class WindowClassification:
    issue: pd.Timestamp
    forecast_end: pd.Timestamp
    eligibility_code: str
    onset_episode_id: str | None
    occurrence_episode_ids: tuple[str, ...]
    interval_occurrence_label: int
    recorded_standard_label: int | None = None

    @property
    def standard_label_matches_intervals(self) -> bool | None:
        if self.recorded_standard_label is None:
            return None
        return int(self.recorded_standard_label) == int(self.interval_occurrence_label)


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def classify_window_from_episode_intervals(
    issue: pd.Timestamp | str,
    forecast_end: pd.Timestamp | str,
    episodes: Sequence[OperationalEpisode],
    *,
    recorded_standard_label: int | None = None,
) -> WindowClassification:
    """Classify a fixed forecast opportunity without using model predictions.

    Boundary semantics are frozen by the mechanism-decomposition contract:
    an episode active at issue time is persistence; a new onset must start
    strictly after issue and strictly before forecast_end; forecast_end is
    exclusive to match the published SEPNET construction.
    """
    issue = _utc(issue)
    forecast_end = _utc(forecast_end)
    if forecast_end <= issue:
        raise ValueError("forecast_end must be after issue")
    if recorded_standard_label not in (None, 0, 1):
        raise ValueError("recorded_standard_label must be None, 0 or 1")

    checked: list[OperationalEpisode] = []
    for ep in episodes:
        ep.validate()
        checked.append(ep)

    active = [ep for ep in checked if ep.start <= issue < ep.end]
    future_starts = [ep for ep in checked if issue < ep.start < forecast_end]
    overlaps = [ep for ep in checked if ep.start < forecast_end and ep.end > issue]

    if len(active) > 1 or len(future_starts) > 1:
        code = "DUPLICATE_OR_AMBIGUOUS"
        onset_id = None
    elif len(active) == 1:
        code = "ALREADY_ACTIVE_PERSISTENCE"
        onset_id = None
    elif len(future_starts) == 1:
        code = "ELIGIBLE_ONSET_POSITIVE"
        onset_id = future_starts[0].episode_id
    else:
        code = "ELIGIBLE_ONSET_NEGATIVE"
        onset_id = None

    return WindowClassification(
        issue=issue,
        forecast_end=forecast_end,
        eligibility_code=code,
        onset_episode_id=onset_id,
        occurrence_episode_ids=tuple(ep.episode_id for ep in overlaps),
        interval_occurrence_label=int(bool(overlaps)),
        recorded_standard_label=recorded_standard_label,
    )


def episode_normalized_occurrence_weights(
    occurrence_episode_ids: Sequence[str | None],
    labels: Sequence[int],
) -> np.ndarray:
    """Weight occurrence positives so each uniquely mapped episode totals one.

    Positive rows without exactly one episode mapping must be passed as None and
    receive zero weight in the normalized-occurrence view. Negatives keep unit
    weight. This preserves the ordinary standard view separately while refusing
    to invent an episode identity for ambiguous positives.
    """
    if len(occurrence_episode_ids) != len(labels):
        raise ValueError("length mismatch")
    y = np.asarray(labels, dtype=int)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary")

    counts: dict[str, int] = {}
    for ep_id, label in zip(occurrence_episode_ids, y, strict=True):
        if label == 1 and ep_id is not None:
            counts[ep_id] = counts.get(ep_id, 0) + 1

    out = np.zeros(len(y), dtype=float)
    for i, (ep_id, label) in enumerate(zip(occurrence_episode_ids, y, strict=True)):
        if label == 0:
            out[i] = 1.0
        elif ep_id is not None:
            out[i] = 1.0 / counts[ep_id]
    return out


def build_threshold_episodes(
    times: Iterable[pd.Timestamp | str],
    flux: Iterable[float],
    *,
    threshold: float = 10.0,
    maximum_gap_minutes: float = 5.0,
    id_prefix: str = "EP",
) -> list[OperationalEpisode]:
    """Construct threshold-exceedance episodes from a fully ordered flux series.

    Non-finite samples and gaps larger than maximum_gap_minutes terminate an
    episode. This function does not interpolate across missing observations.
    """
    t = pd.DatetimeIndex(pd.to_datetime(list(times), utc=True))
    v = np.asarray(list(flux), dtype=float)
    if len(t) != len(v):
        raise ValueError("times/flux length mismatch")
    if len(t) == 0:
        return []
    ns = t.asi8
    if np.any(np.diff(ns) <= 0):
        raise ValueError("timestamps must be strictly increasing and unique")
    max_gap_ns = int(maximum_gap_minutes * 60 * 1e9)

    episodes: list[OperationalEpisode] = []
    start_idx: int | None = None
    episode_counter = 0
    for i in range(len(v)):
        valid = np.isfinite(v[i]) and v[i] >= 0
        above = valid and v[i] >= threshold
        gap_from_previous = i > 0 and ns[i] - ns[i - 1] > max_gap_ns

        if start_idx is not None and (not above or gap_from_previous):
            end_idx = i - 1
            # End is one cadence after last confirmed above-threshold sample when
            # the next sample is contiguous; otherwise it is exactly last sample.
            if i < len(v) and not gap_from_previous:
                end_time = t[i]
            else:
                end_time = t[end_idx] + pd.Timedelta(minutes=maximum_gap_minutes)
            episode_counter += 1
            episodes.append(
                OperationalEpisode(
                    f"{id_prefix}{episode_counter:05d}",
                    pd.Timestamp(t[start_idx]),
                    pd.Timestamp(end_time),
                )
            )
            start_idx = None

        if start_idx is None and above:
            start_idx = i

    if start_idx is not None:
        episode_counter += 1
        episodes.append(
            OperationalEpisode(
                f"{id_prefix}{episode_counter:05d}",
                pd.Timestamp(t[start_idx]),
                pd.Timestamp(t[-1] + pd.Timedelta(minutes=maximum_gap_minutes)),
            )
        )
    return episodes


def attrition_counts(codes: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for code in codes:
        out[str(code)] = out.get(str(code), 0) + 1
    if sum(out.values()) != len(codes):
        raise AssertionError("attrition ledger failed to reconcile")
    return dict(sorted(out.items()))
