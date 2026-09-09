"""Exact parsing of JSOC ``_TAI`` record timestamps.

JSOC record keys such as ``2017.01.01_00:00:00_TAI`` are expressed in the
International Atomic Time scale.  Treating the text as UTC is usually invisible
in an image-level inspection, but it can move a record across an hourly join
boundary.  Astropy owns the leap-second table, so this module is the single
conversion point used by both manifest construction and model data loading.
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


_JSOC_TAI = re.compile(
    r"^(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})_"
    r"(?P<clock>\d{2}:\d{2}:\d{2}(?:\.\d+)?)_TAI$"
)


def parse_jsoc_trec_to_utc(values: Iterable[object], *, require_tai: bool = True) -> pd.Series:
    """Return a timezone-aware UTC series from JSOC ``T_REC`` values.

    Parameters
    ----------
    values:
        An iterable or :class:`pandas.Series` of JSOC timestamp strings.
    require_tai:
        Require the explicit ``_TAI`` suffix.  Production manifest construction
        uses the default because silently guessing a time scale is unsafe.  A
        caller handling a separately documented UTC export may set this to
        ``False``; non-TAI values are then parsed as UTC.

    Invalid values become ``NaT``.  If ``require_tai`` is true, a non-empty value
    with no valid JSOC TAI form raises before any join can proceed.
    """

    raw = pd.Series(values, copy=False).astype("string").str.strip()
    nonempty = raw.notna() & raw.ne("") & raw.ne("<NA>") & raw.ne("nan")
    match = raw.str.extract(_JSOC_TAI, expand=False)
    tai_mask = nonempty & match.year.notna()

    if require_tai:
        bad = nonempty & ~tai_mask
        if bool(bad.any()):
            sample = raw[bad].head(3).tolist()
            raise ValueError(f"Expected explicit JSOC _TAI T_REC values; invalid examples: {sample}")

    out = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    if bool(tai_mask.any()):
        try:
            from astropy.time import Time
        except ImportError as exc:  # pragma: no cover - exercised in minimal envs
            raise ImportError("Astropy is required for exact JSOC TAI-to-UTC conversion") from exc

        parts = match.loc[tai_mask]
        isot = (
            parts["year"]
            + "-"
            + parts["month"]
            + "-"
            + parts["day"]
            + "T"
            + parts["clock"]
        )
        converted = Time(isot.tolist(), format="isot", scale="tai").utc.to_datetime()
        out.loc[tai_mask] = pd.to_datetime(converted, utc=True)

    utc_mask = nonempty & ~tai_mask
    if bool(utc_mask.any()):
        out.loc[utc_mask] = pd.to_datetime(raw.loc[utc_mask], utc=True, errors="coerce")
    return out
