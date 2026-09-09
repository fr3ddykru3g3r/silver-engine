"""Causal 24-hour feature aggregation for the frozen IRIS-SEP interface.

This module reproduces the *aggregation semantics* of the pinned upstream
``Data_Aggregation.R`` while deliberately refusing its retrospective repair
steps. Inputs are already-observed source rows captured no later than forecast
issue time. No nearest-future fill, overlap backcast, CME regression fill, or
runtime reconstruction occurs here.

The output is checked against the exact frozen package feature lists. A missing
expected column is an error; an unavailable native measurement remains NaN (or
the explicit upstream empty-window constant where the original recipe used one).
"""
from __future__ import annotations

from datetime import datetime, timedelta
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


WINDOW = timedelta(hours=24)
FLARE_LOG_STRENGTH_EMPTY = math.log10(1e-10)
_STATS = ("min", "max", "avg")


class CausalFeatureAggregationError(ValueError):
    """Raised when a causal source snapshot cannot form the frozen feature row."""


def _aware(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CausalFeatureAggregationError(f"{name} must be timezone-aware")
    return value


def _time_column(frame: pd.DataFrame, candidates: Sequence[str], *, name: str) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        raise CausalFeatureAggregationError(f"{name} source must be a DataFrame")
    column = next((value for value in candidates if value in frame.columns), None)
    if column is None:
        if len(frame) == 0:
            return pd.Series([], dtype="datetime64[ns, UTC]")
        raise CausalFeatureAggregationError(f"{name} source has no recognized timestamp column")
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if bool(values.isna().any()):
        raise CausalFeatureAggregationError(f"{name} source has invalid timestamps")
    return values


def _window(frame: pd.DataFrame, times: pd.Series, *, issue_time: datetime, name: str) -> pd.DataFrame:
    if len(frame) != len(times):
        raise CausalFeatureAggregationError(f"{name} time alignment failed")
    if len(times) and bool((times > issue_time).any()):
        raise CausalFeatureAggregationError(f"{name} source contains observations after issue time")
    start = issue_time - WINDOW
    mask = (times >= start) & (times < issue_time)
    return frame.loc[mask].copy().reset_index(drop=True)


def _numeric_stats(values: pd.Series | Sequence[Any], *, empty_value: float = math.nan) -> dict[str, float]:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))]
    if len(finite) == 0:
        return {"min": float(empty_value), "max": float(empty_value), "avg": float(empty_value)}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "avg": float(finite.mean()),
    }


def _assign_stats(row: dict[str, float | int], prefix: str, values, *, empty_value: float = math.nan) -> None:
    stats = _numeric_stats(values, empty_value=empty_value)
    for suffix in _STATS:
        row[f"{prefix}_{suffix}"] = stats[suffix]


def _sharp_feature_names(solar_features: Sequence[str]) -> tuple[str, ...]:
    direct = set()
    ar = set()
    direct_re = re.compile(r"^SHARP_(.+)_(min|max|avg)$")
    ar_re = re.compile(r"^SHARP_AR_(.+)_(min|max|avg)$")
    for name in solar_features:
        m_ar = ar_re.match(name)
        if m_ar:
            ar.add(m_ar.group(1))
            continue
        m = direct_re.match(name)
        if m:
            direct.add(m.group(1))
    if direct != ar:
        missing_ar = sorted(direct - ar)
        missing_direct = sorted(ar - direct)
        raise CausalFeatureAggregationError(
            f"SHARP/SHARP_AR frozen feature sets differ; missing_ar={missing_ar[:5]}, missing_direct={missing_direct[:5]}"
        )
    if not direct:
        raise CausalFeatureAggregationError("no SHARP features found in frozen solar schema")
    return tuple(sorted(direct))


def _goes_class_flux(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    text = str(value).replace(",", ".").upper()
    text = re.sub(r"[^ABCMXSF0-9.]", "", text)
    if text == "SF":
        text = "C"
    match = re.fullmatch(r"([ABCMX])([0-9.]*)", text)
    if not match:
        return math.nan
    magnitude = 1.0
    if match.group(2):
        try:
            magnitude = float(match.group(2))
        except ValueError:
            magnitude = 1.0
        if not math.isfinite(magnitude) or magnitude <= 0:
            magnitude = 1.0
    base = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[match.group(1)]
    return magnitude * base


def _flare_features(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) == 0:
        return pd.DataFrame(columns=["Duration", "RisenTime", "log_Strength", "ar_noaanum"])
    required = {"start_time", "peak_time", "end_time", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CausalFeatureAggregationError(f"flare source missing fields: {missing}")
    out = frame.copy()
    start = pd.to_datetime(out["start_time"], utc=True, errors="coerce")
    peak = pd.to_datetime(out["peak_time"], utc=True, errors="coerce")
    end = pd.to_datetime(out["end_time"], utc=True, errors="coerce")
    if bool(start.isna().any() or peak.isna().any() or end.isna().any()):
        raise CausalFeatureAggregationError("flare source has invalid start/peak/end timestamps")
    swap = peak > end
    peak2 = peak.copy()
    end2 = end.copy()
    peak2.loc[swap], end2.loc[swap] = end.loc[swap], peak.loc[swap]
    # Legacy upstream repair: if peak < start, move peak to midnight of the end date.
    bad = peak2 < start
    if bool(bad.any()):
        midnight = end2.loc[bad].dt.floor("D")
        peak2.loc[bad] = midnight
    if bool(((peak2 < start) | (peak2 > end2)).any()):
        raise CausalFeatureAggregationError("flare peak remains inconsistent after frozen repair rule")
    out["Duration"] = (end2 - start).dt.total_seconds() / 60.0
    out["RisenTime"] = (peak2 - start).dt.total_seconds() / 60.0
    flux = out["label"].map(_goes_class_flux).astype(float)
    out["log_Strength"] = np.log10(flux)
    if "ar_noaanum" not in out.columns:
        out["ar_noaanum"] = np.nan
    return out


def _fill_sharp(row: dict[str, Any], *, sharp: pd.DataFrame, flare_ar: set[int], features: Sequence[str]) -> None:
    row["SHARP_label"] = int(len(sharp) > 0)
    row["SHARP_From_SMARP_label"] = 0
    for feature in features:
        if len(sharp) and feature not in sharp.columns:
            raise CausalFeatureAggregationError(f"native SHARP source missing frozen feature {feature}")
        values = sharp[feature] if len(sharp) else []
        stats = _numeric_stats(values)
        for suffix in _STATS:
            row[f"SHARP_{feature}_{suffix}"] = stats[suffix]

    if len(sharp) and "NOAA_AR" not in sharp.columns:
        raise CausalFeatureAggregationError("native SHARP source missing NOAA_AR")
    if flare_ar and len(sharp):
        noaa = pd.to_numeric(sharp["NOAA_AR"], errors="coerce")
        selected = sharp.loc[noaa.isin(flare_ar)].copy()
    else:
        selected = sharp.iloc[0:0].copy()
    row["SHARP_AR_label"] = int(len(selected) > 0)
    row["SHARP_AR_From_SMARP_label"] = 0
    for feature in features:
        stats = _numeric_stats(selected[feature] if len(selected) else [])
        for suffix in _STATS:
            row[f"SHARP_AR_{feature}_{suffix}"] = stats[suffix]


def _fill_flares(row: dict[str, Any], flare: pd.DataFrame) -> None:
    if len(flare) == 0:
        row["Flare_label"] = 0
        row["Flare_num"] = 0
        for feature in ("Duration", "RisenTime"):
            for suffix in _STATS:
                row[f"Flare_{feature}_{suffix}"] = 0.0
        for suffix in _STATS:
            row[f"Flare_log_Strength_{suffix}"] = FLARE_LOG_STRENGTH_EMPTY
        return
    row["Flare_label"] = 1
    row["Flare_num"] = int(len(flare))
    for feature in ("Duration", "RisenTime", "log_Strength"):
        stats = _numeric_stats(flare[feature])
        for suffix in _STATS:
            row[f"Flare_{feature}_{suffix}"] = stats[suffix]


def _fill_donki(row: dict[str, Any], cme: pd.DataFrame) -> None:
    row["DONKICME_label"] = int(len(cme) > 0)
    row["DONKICME_num"] = int(len(cme))
    # Prospective causal pipeline never synthesizes DONKI from CDAW.
    row["DONKICME_From_CDAW_label"] = 0
    for feature in ("lon_deg", "halfAngle_deg", "speed_km_s"):
        if len(cme) and feature not in cme.columns:
            raise CausalFeatureAggregationError(f"DONKI source missing {feature}")
        stats = _numeric_stats(cme[feature] if len(cme) else [], empty_value=0.0)
        for suffix in _STATS:
            row[f"DONKICME_{feature}_{suffix}"] = stats[suffix]


def _fill_cdaw(row: dict[str, Any], cme: pd.DataFrame) -> None:
    row["CDAWCME_label"] = int(len(cme) > 0)
    row["CDAWCME_num"] = int(len(cme))
    for feature in ("central_pa_deg", "width_deg", "speed_km_s", "energy_erg"):
        if len(cme) and feature not in cme.columns:
            raise CausalFeatureAggregationError(f"CDAW source missing {feature}")
        stats = _numeric_stats(cme[feature] if len(cme) else [], empty_value=0.0)
        for suffix in _STATS:
            row[f"CDAWCME_{feature}_{suffix}"] = stats[suffix]


def _fill_scalar_family(row: dict[str, Any], *, prefix: str, label_name: str, values) -> None:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series.to_numpy(dtype=float, na_value=np.nan))]
    row[label_name] = int(len(finite) > 0)
    stats = _numeric_stats(finite if len(finite) else [0.0], empty_value=0.0)
    for suffix in _STATS:
        row[f"{prefix}_{suffix}"] = stats[suffix]


def build_causal_feature_row(
    *,
    issue_time: datetime,
    feature_families: Mapping[str, Sequence[str]],
    sharp: pd.DataFrame,
    flares: pd.DataFrame,
    donki_cme: pd.DataFrame,
    cdaw_cme: pd.DataFrame,
    proton: pd.DataFrame,
    xrs: pd.DataFrame,
) -> pd.DataFrame:
    """Return exactly one frozen-schema feature row from past-only source data."""
    issue = _aware(issue_time, "issue_time")
    if set(feature_families) != {"solar", "xrs", "proton"}:
        raise CausalFeatureAggregationError("feature_families must be exactly solar/xrs/proton")

    sharp_win = _window(sharp, _time_column(sharp, ("T_REC", "T_REC_posix"), name="SHARP"), issue_time=issue, name="SHARP")
    flare_peak = _time_column(flares, ("peak_time", "event_peaktime"), name="flare")
    flare_win_raw = _window(flares, flare_peak, issue_time=issue, name="flare")
    flare_win = _flare_features(flare_win_raw)
    donki_win = _window(donki_cme, _time_column(donki_cme, ("startTime", "cme_time", "cme_time_posix"), name="DONKI"), issue_time=issue, name="DONKI")
    cdaw_win = _window(cdaw_cme, _time_column(cdaw_cme, ("cme_time", "cme_time_posix"), name="CDAW"), issue_time=issue, name="CDAW")
    proton_win = _window(proton, _time_column(proton, ("time", "time_tag", "time_posix"), name="proton"), issue_time=issue, name="proton")
    xrs_win = _window(xrs, _time_column(xrs, ("time", "time_tag", "time_posix"), name="XRS"), issue_time=issue, name="XRS")

    flare_ar: set[int] = set()
    if len(flare_win):
        ar = pd.to_numeric(flare_win["ar_noaanum"], errors="coerce").dropna()
        flare_ar = {int(value) for value in ar if int(value) > 0}

    row: dict[str, Any] = {}
    _fill_sharp(row, sharp=sharp_win, flare_ar=flare_ar, features=_sharp_feature_names(feature_families["solar"]))
    _fill_flares(row, flare_win)
    _fill_donki(row, donki_win)
    _fill_cdaw(row, cdaw_win)

    proton_value = next((name for name in ("ProtonFlux", "flux") if name in proton_win.columns), None)
    if proton_value is None and len(proton_win):
        raise CausalFeatureAggregationError("proton source missing ProtonFlux/flux")
    _fill_scalar_family(
        row,
        prefix="ProtonFlux",
        label_name="ProtonFlux_label",
        values=proton_win[proton_value] if proton_value else [],
    )

    xrs_value = next((name for name in ("xrsb", "flux") if name in xrs_win.columns), None)
    if xrs_value is None and len(xrs_win):
        raise CausalFeatureAggregationError("XRS source missing xrsb/flux")
    _fill_scalar_family(
        row,
        prefix="XRSB",
        label_name="XRS_label",
        values=xrs_win[xrs_value] if xrs_value else [],
    )

    expected = [name for family in ("solar", "xrs", "proton") for name in feature_families[family]]
    missing = [name for name in expected if name not in row]
    unexpected = sorted(set(row) - set(expected))
    if missing or unexpected:
        raise CausalFeatureAggregationError(
            f"causal feature row does not match frozen schema; missing={missing[:12]}, unexpected={unexpected[:12]}"
        )
    return pd.DataFrame([[row[name] for name in expected]], columns=expected)
