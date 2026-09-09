"""Compatibility wrapper for the frozen freshness crossover study.

Adds bounded HTTP retries and only representation-level fixes for pandas
timezone-aware timestamps. Scientific choices in the frozen runner are unchanged:
URLs, labels, features, models, roles, thresholds, delays and evaluation rules.
No model scores had been produced before these compatibility fixes were frozen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_OriginalSession = requests.Session


class ReliableSession(_OriginalSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        retry = Retry(
            total=7,
            connect=7,
            read=7,
            status=7,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self.mount("https://", adapter)
        self.mount("http://", adapter)


requests.Session = ReliableSession

from iris_report.iris_sep.tools import run_freshness_crossover_study_v1 as m


def _asi8(values) -> np.ndarray:
    """Return UTC nanoseconds for tz-aware pandas timestamp-like values."""
    return pd.DatetimeIndex(values).asi8


def robust_idx(times, issue, delay):
    ns = _asi8(times)
    ins = issue.value
    lo = ins - int(24 * 3600 * 1e9)
    cutoff = min(ins - 1, ins - int(delay * 60 * 1e9))
    return int(np.searchsorted(ns, lo, "left")), int(np.searchsorted(ns, cutoff, "right"))


def robust_family_features(df, issue, delay, family):
    a, b = robust_idx(df["time"], issue, delay)
    sub = df.iloc[a:b]
    tns = _asi8(sub["time"])
    ins = issue.value
    if family == "proton":
        v = sub["proton"].to_numpy(float)
        out = m.stream_features(tns, v, ins, 288, "p", True)
        good = v[np.isfinite(v)]
        for q in m.P_THRESH:
            out[f"p_count_ge_{q:g}"] = float(np.sum(good >= q))
        return out
    out = m.stream_features(tns, sub["A"].to_numpy(float), ins, 1440, "a")
    out.update(m.stream_features(tns, sub["B"].to_numpy(float), ins, 1440, "b"))
    good = sub["B"].to_numpy(float)
    good = good[np.isfinite(good)]
    for q in m.XRS_THRESH:
        out[f"b_count_ge_{q:.0e}"] = float(np.sum(good >= q))
    return out


def robust_label_issue(proton, issue):
    t = _asi8(proton["time"])
    v = proton["proton"].to_numpy(float)
    ins = issue.value
    j = np.searchsorted(t, ins, "right") - 1
    if j < 0 or not np.isfinite(v[j]) or ins - t[j] > 5 * 60e9:
        return {"resolved": False, "active": False, "label": None, "crossing_time": None, "reason": "issue_support"}
    if v[j] >= 10:
        return {"resolved": True, "active": True, "label": None, "crossing_time": None, "reason": "already_active"}
    end = ins + 24 * 3600 * 1e9
    k = np.searchsorted(t, end, "left")
    if k >= len(t) or t[k] != end or not np.isfinite(v[k]):
        return {"resolved": False, "active": False, "label": None, "crossing_time": None, "reason": "endpoint"}
    segt = t[j : k + 1]
    segv = v[j : k + 1]
    if np.any(~np.isfinite(segv)) or np.any(np.diff(segt) > 5 * 60e9):
        return {"resolved": False, "active": False, "label": None, "crossing_time": None, "reason": "gap_or_invalid"}
    cross = None
    for q in range(1, len(segv)):
        if segt[q] > ins and segv[q - 1] < 10 <= segv[q]:
            cross = pd.to_datetime(int(segt[q]), unit="ns", utc=True)
            break
    return {"resolved": True, "active": False, "label": int(cross is not None), "crossing_time": cross, "reason": "ok"}


def robust_bootstrap_delay(roles, store, info, reps=2000):
    rng = np.random.default_rng(20260909)
    out = []
    base = roles["score"].reset_index(drop=True)
    y = base.y.to_numpy()
    episode = base.crossing_time.astype(str).to_numpy()
    quiet_ns = _asi8(base.issue.dt.floor("D"))
    quiet = quiet_ns // int(7 * 86400 * 1e9)
    pos_groups = [np.where((y == 1) & (episode == e))[0] for e in np.unique(episode[y == 1])]
    neg_groups = [np.where((y == 0) & (quiet == q))[0] for q in np.unique(quiet[y == 0])]
    if not pos_groups or not neg_groups:
        raise RuntimeError("paired bootstrap requires at least one positive episode and one quiet block")
    for fam in ["xrs", "proton"]:
        for d in m.DELAYS:
            ps = store[("score", fam, d)][0]
            for method in ["joint", "reduced", "delayed_single", "clean_joint"]:
                vals = []
                th = m.threshold_for(method, fam, info)
                for _ in range(reps):
                    ix = np.concatenate(
                        [pos_groups[i] for i in rng.integers(0, len(pos_groups), len(pos_groups))]
                        + [neg_groups[i] for i in rng.integers(0, len(neg_groups), len(neg_groups))]
                    )
                    vals.append(m.tss(m.counts(y[ix], ps[method][ix], th)))
                out.append(
                    {
                        "delayed_family": fam,
                        "delay_minutes": d,
                        "method": method,
                        "TSS_ci_low": float(np.nanpercentile(vals, 2.5)),
                        "TSS_ci_high": float(np.nanpercentile(vals, 97.5)),
                    }
                )
            dif = []
            tj = info["joint"]["threshold"]
            tr = info["proton" if fam == "xrs" else "xrs"]["threshold"]
            for _ in range(reps):
                ix = np.concatenate(
                    [pos_groups[i] for i in rng.integers(0, len(pos_groups), len(pos_groups))]
                    + [neg_groups[i] for i in rng.integers(0, len(neg_groups), len(neg_groups))]
                )
                dif.append(
                    m.tss(m.counts(y[ix], ps["reduced"][ix], tr))
                    - m.tss(m.counts(y[ix], ps["joint"][ix], tj))
                )
            out.append(
                {
                    "delayed_family": fam,
                    "delay_minutes": d,
                    "method": "reduced_minus_joint",
                    "TSS_ci_low": float(np.nanpercentile(dif, 2.5)),
                    "TSS_ci_high": float(np.nanpercentile(dif, 97.5)),
                }
            )
    return pd.DataFrame(out)


m.idx = robust_idx
m.family_features = robust_family_features
m.label_issue = robust_label_issue
m.bootstrap_delay = robust_bootstrap_delay
m.main()
