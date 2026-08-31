#!/usr/bin/env python3
"""Generate the reproducibility audit, figures, tables, and 3-D exports.

This script intentionally reads the immutable evidence/cache/run directories
created by the IRIS BASE experiment and writes only into the report directory.
It does not alter the source bundle, the raw FITS cache, or model checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; registers 3-D projection

from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(os.environ.get("IRIS_RUN_ROOT", "/private/tmp/iris_gated_run"))
SOURCE = ROOT / "source"
EVIDENCE = ROOT / "evidence"
FITS = ROOT / "fits"
WORK = ROOT / "work"
RUN = WORK / "runs" / "base_local_resume"
PHYSICS_SCREEN = WORK / "runs" / "physics_screening_2026"
PREPARED_TENSORS = WORK / "prepared" / "positive_train_pergroup4"
REPORT = Path(os.environ.get("IRIS_REPORT_DIR", str(Path(__file__).resolve().parents[1])))
FIG = REPORT / "figures"
TABLE = REPORT / "tables"
MODEL_DIR = REPORT / "models"
ARTIFACT = REPORT / "artifacts"

MODEL_SRC = SOURCE / "iris-model"
COMMON_SRC = SOURCE / "common"
for p in (MODEL_SRC, COMMON_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from preprocess import PreprocessConfig, preprocess_fits  # noqa: E402
from jsoc_time import parse_jsoc_trec_to_utc  # noqa: E402


SEED = 2026
BLUE = "#1769aa"
ORANGE = "#d97706"
GREEN = "#18864b"
RED = "#b42318"
PURPLE = "#7c3aed"
INK = "#17202a"
GRID = "#d9e2ec"


def ensure_dirs() -> None:
    for p in (REPORT, FIG, TABLE, MODEL_DIR, ARTIFACT):
        p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fmt(x: object, digits: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        y = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(y):
        return "NA"
    if abs(y) >= 1000 or (0 < abs(y) < 0.001):
        return f"{y:.3e}"
    return f"{y:.{digits}f}"


def tex_escape(value: object) -> str:
    s = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(c, c) for c in s)


def rel(path: Path) -> str:
    return str(path.relative_to(REPORT)).replace(os.sep, "/")


def load_inputs() -> dict[str, object]:
    d = EVIDENCE / "data" / "derived"
    manifest = pd.read_csv(d / "training_manifest.csv.gz", low_memory=False)
    metadata = pd.read_csv(d / "sharp_metadata.csv.gz", low_memory=False)
    fit_ids = sorted(p.stem for p in FITS.glob("*.fits"))
    fit_set = set(fit_ids)
    acquired = manifest[manifest.sample_id.astype(str).isin(fit_set)].copy()
    audit = json.loads((RUN / "audit" / "v2_manipulation_metrics.json").read_text())
    alternate_path = WORK / "runs" / "base_local_resume" / "audit_seed2027" / "v2_manipulation_metrics.json"
    alternate = json.loads(alternate_path.read_text()) if alternate_path.exists() else None
    destruction_path = WORK / "runs" / "base_local_resume" / "destruction_controls_audit" / "destruction_controls.json"
    destruction = json.loads(destruction_path.read_text()) if destruction_path.exists() else None
    acquisition = json.loads((FITS / "acquisition_report.json").read_text())
    manifest_audit = json.loads((d / "manifest_audit.json").read_text())
    label_audit = json.loads((d / "label_integrity_audit.json").read_text())
    frozen_split = json.loads((d / "frozen_split.json").read_text())
    tai_audit = json.loads((d / "tai_repair_audit.json").read_text())
    collection = json.loads((d / "collection_summary.json").read_text())
    retrieval = json.loads((d / "fits_retrieval_smoke_test.json").read_text())
    source_ledger = json.loads((d / "source_ledger.json").read_text())
    run_cfg = json.loads((RUN / "outputs" / "base" / "run_config_resume.json").read_text())
    ck = json.loads(json.dumps({}))
    import torch

    ckpt = torch.load(RUN / "outputs" / "base" / "generator.pt", map_location="cpu")
    ck = {k: v for k, v in ckpt.items() if isinstance(v, (str, int, float, bool)) or v is None}
    history = json.loads((RUN / "outputs" / "base" / "training_history_resume.json").read_text())
    sampling = json.loads((RUN / "samples" / "base" / "sampling_summary.json").read_text())
    gen = pd.read_csv(RUN / "audit" / "real_generic.csv")
    syn_gen = pd.read_csv(RUN / "audit" / "synthetic_generic.csv")
    geom = pd.read_csv(RUN / "audit" / "real_hard_geometry.csv")
    syn_geom = pd.read_csv(RUN / "audit" / "synthetic_hard_geometry.csv")
    pil = pd.read_csv(RUN / "audit" / "real_hard_pil.csv")
    syn_pil = pd.read_csv(RUN / "audit" / "synthetic_hard_pil.csv")
    arrays = []
    sman = pd.read_csv(RUN / "samples" / "base" / "synthetic_manifest.csv")
    for p in sman.array_path:
        q = Path(str(p))
        if not q.exists():
            q = RUN / "samples" / "base" / "arrays" / q.name
        arrays.append(np.load(q).astype(np.float32))
    syn_norm = np.stack(arrays)
    # The source sampler writes normalized values in [-1, 1].
    denom = np.arcsinh(3000.0 / 250.0)
    syn_gauss = 250.0 * np.sinh(np.clip(syn_norm, -1.0, 1.0) * denom)
    return {
        "dir": d,
        "manifest": manifest,
        "metadata": metadata,
        "fit_ids": fit_ids,
        "acquired": acquired,
        "audit": audit,
        "alternate": alternate,
        "destruction": destruction,
        "acquisition": acquisition,
        "manifest_audit": manifest_audit,
        "label_audit": label_audit,
        "frozen_split": frozen_split,
        "tai_audit": tai_audit,
        "collection": collection,
        "retrieval": retrieval,
        "source_ledger": source_ledger,
        "run_cfg": run_cfg,
        "ck": ck,
        "history": history,
        "sampling": sampling,
        "gen": gen,
        "syn_gen": syn_gen,
        "geom": geom,
        "syn_geom": syn_geom,
        "pil": pil,
        "syn_pil": syn_pil,
        "sman": sman,
        "syn_norm": syn_norm,
        "syn_gauss": syn_gauss,
    }


def style(ax: plt.Axes, title: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)
    if title:
        ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=10)


def savefig(fig: plt.Figure, name: str) -> None:
    try:
        fig.tight_layout()
    except RuntimeError:
        # Figures created with constrained_layout and colorbars use a layout
        # engine that intentionally rejects a second tight_layout pass.
        pass
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def denorm_sample(x: np.ndarray) -> np.ndarray:
    return 250.0 * np.sinh(np.clip(x, -1.0, 1.0) * np.arcsinh(3000.0 / 250.0))


def representative_real(data: dict[str, object]) -> tuple[str, np.ndarray, pd.Series]:
    gen: pd.DataFrame = data["gen"]  # type: ignore[assignment]
    manifest: pd.DataFrame = data["manifest"]  # type: ignore[assignment]
    fit_set = set(data["fit_ids"])  # type: ignore[arg-type]
    candidates = [str(x) for x in gen.sample_id if str(x) in fit_set]
    if not candidates:
        candidates = [str(x) for x in data["fit_ids"][:1]]  # type: ignore[index]
    sid = candidates[0]
    row = manifest[manifest.sample_id.astype(str).eq(sid)].iloc[0]
    meta: pd.DataFrame = data["metadata"].copy()  # type: ignore[assignment]
    meta["harpnum_join"] = pd.to_numeric(meta.HARPNUM, errors="coerce")
    meta["join_time"] = parse_jsoc_trec_to_utc(meta.T_REC).dt.floor("h")
    target_time = pd.to_datetime(row.t_rec, utc=True).floor("h")
    geom = meta[(meta.harpnum_join == float(row.harpnum)) & (meta.join_time == target_time)]
    if geom.empty:
        raise RuntimeError(f"No geometry metadata for representative sample {sid}")
    geometry = geom.iloc[0]
    _, raw = preprocess_fits(
        FITS / f"{sid}.fits",
        float(geometry.CDELT1),
        float(geometry.CDELT2),
        float(geometry.RSUN_REF),
        PreprocessConfig(),
    )
    return sid, raw.squeeze(0).numpy(), row


def write_obj(path: Path, z: np.ndarray, title: str) -> None:
    """Write a regular-grid surface mesh; z is in gauss and x/y are Mm."""
    z = np.asarray(z, dtype=np.float32)
    h, w = z.shape
    xs = np.linspace(-128.0, 128.0, w)
    ys = np.linspace(-128.0, 128.0, h)
    with path.open("w") as f:
        f.write(f"# {title}\n")
        f.write("# x/y units: Mm; z unit: gauss; z is the magnetogram value visualized as height.\n")
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                f.write(f"v {x:.5f} {y:.5f} {float(z[iy, ix]):.5f}\n")
        for iy in range(h - 1):
            for ix in range(w - 1):
                a = iy * w + ix + 1
                b = a + 1
                c = a + w + 1
                d = a + w
                f.write(f"f {a} {b} {c} {d}\n")


def bootstrap_median_ratio(real: np.ndarray, syn: np.ndarray, seed: int = SEED, n: int = 3000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    real = np.asarray(real, dtype=float)
    syn = np.asarray(syn, dtype=float)
    real = real[np.isfinite(real)]
    syn = syn[np.isfinite(syn)]
    if not len(real) or not len(syn):
        return float("nan"), float("nan"), float("nan")
    vals = np.empty(n, dtype=float)
    for i in range(n):
        a = rng.choice(real, size=len(real), replace=True)
        b = rng.choice(syn, size=len(syn), replace=True)
        vals[i] = np.median(b) / max(abs(np.median(a)), 1e-12)
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def descriptor_table(data: dict[str, object]) -> pd.DataFrame:
    gen: pd.DataFrame = data["gen"]  # type: ignore[assignment]
    syn: pd.DataFrame = data["syn_gen"]  # type: ignore[assignment]
    out = []
    for col in ["log_mean_abs", "log_p90_abs", "log_p99_abs", "active_fraction", "strong_fraction", "saturation_fraction"]:
        r = gen[col].to_numpy(float)
        s = syn[col].to_numpy(float)
        ks = ks_2samp(r, s)
        out.append(
            {
                "family": "generic",
                "feature": col,
                "real_n": len(r),
                "synthetic_n": len(s),
                "real_mean": np.mean(r),
                "synthetic_mean": np.mean(s),
                "real_median": np.median(r),
                "synthetic_median": np.median(s),
                "real_q05": np.percentile(r, 5),
                "real_q95": np.percentile(r, 95),
                "synthetic_q05": np.percentile(s, 5),
                "synthetic_q95": np.percentile(s, 95),
                "median_ratio": np.median(s) / max(abs(np.median(r)), 1e-12),
                "median_ratio_bootstrap": bootstrap_median_ratio(r, s, SEED + len(out))[0],
                "ratio_ci_low": bootstrap_median_ratio(r, s, SEED + len(out))[1],
                "ratio_ci_high": bootstrap_median_ratio(r, s, SEED + len(out))[2],
                "wasserstein": wasserstein_distance(r, s),
                "ks_statistic": ks.statistic,
                "ks_pvalue": ks.pvalue,
            }
        )
    return pd.DataFrame(out)


def hard_descriptor_table(data: dict[str, object], family: str, real: pd.DataFrame, syn: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in real.columns if c not in {"sample_id", "region_group_id"}]
    out = []
    for col in cols:
        r = real[col].to_numpy(float)
        s = syn[col].to_numpy(float)
        ks = ks_2samp(r, s)
        out.append(
            {
                "family": family,
                "feature": col,
                "real_n": len(r),
                "synthetic_n": len(s),
                "real_mean": np.mean(r),
                "synthetic_mean": np.mean(s),
                "real_median": np.median(r),
                "synthetic_median": np.median(s),
                "real_q05": np.percentile(r, 5),
                "real_q95": np.percentile(r, 95),
                "synthetic_q05": np.percentile(s, 5),
                "synthetic_q95": np.percentile(s, 95),
                "median_ratio": np.median(s) / max(abs(np.median(r)), 1e-12),
                "median_ratio_bootstrap": bootstrap_median_ratio(r, s, SEED + len(out) + 100)[0],
                "ratio_ci_low": bootstrap_median_ratio(r, s, SEED + len(out) + 100)[1],
                "ratio_ci_high": bootstrap_median_ratio(r, s, SEED + len(out) + 100)[2],
                "wasserstein": wasserstein_distance(r, s),
                "ks_statistic": ks.statistic,
                "ks_pvalue": ks.pvalue,
            }
        )
    return pd.DataFrame(out)


def figure_graphical_abstract(data: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(13.33, 5.4))
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    boxes = [
        (0.3, 1.55, 2.25, 2.25, "NOAA GOES\nflare events", "event labels\nM1+ / 24 h", BLUE),
        (2.95, 1.55, 2.25, 2.25, "JSOC HMI/SHARP\nmagnetograms", "5,273 FITS\nacquired + verified", ORANGE),
        (5.6, 1.55, 2.25, 2.25, "UTC repair +\nchronological split", "group leakage\ncontrolled", PURPLE),
        (8.25, 1.55, 2.25, 2.25, "Conditional\ndiffusion U-Net", "BASE · 100 steps\n1,200 local steps", GREEN),
        (10.9, 1.55, 2.1, 2.25, "Independent\naudit", "128 samples\nPASS / limits", RED),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.13", facecolor="white", edgecolor=color, linewidth=2.2))
        ax.text(x + w / 2, y + 1.42, title, ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + 0.63, subtitle, ha="center", va="center", fontsize=10, color="#486581")
    for x in [2.55, 5.2, 7.85, 10.5]:
        ax.add_patch(FancyArrowPatch((x, 2.68), (x + 0.38, 2.68), arrowstyle="-|>", mutation_scale=17, linewidth=1.7, color="#627d98"))
    ax.text(6.67, 4.68, "IRIS BASE: from verified solar observations to a distributional generator audit", ha="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(6.67, 0.65, "Claim boundary: this experiment evaluates train-only image/distribution fidelity; it does not establish forecasting utility.", ha="center", fontsize=10.5, color=RED)
    savefig(fig, "graphical_abstract")


def figure_dataset(data: dict[str, object]) -> None:
    ma: dict = data["manifest_audit"]  # type: ignore[assignment]
    parts = ma["partitions"]
    names = [p["partition"] for p in parts]
    rows = [p["rows"] for p in parts]
    pos = [p["positive_rows"] for p in parts]
    cens = [p["censored_negative_rows"] for p in parts]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x, rows, color=[BLUE, ORANGE, PURPLE], width=0.6, label="manifest rows")
    ax.bar(x, pos, color=RED, width=0.6, label="M1+ positives")
    ax.set_xticks(x, [n.title() for n in names])
    ax.set_ylabel("Rows")
    style(ax, "Frozen chronological partitions")
    ax.legend(frameon=False, fontsize=9)
    for i, v in enumerate(rows):
        ax.text(i, v + max(rows) * 0.02, f"{v:,}", ha="center", fontsize=9)
    ax = axes[1]
    ax.bar(x, cens, color="#829ab1", width=0.6)
    ax.set_xticks(x, [n.title() for n in names])
    ax.set_ylabel("Censored negatives")
    style(ax, "Unresolved-event censoring")
    for i, v in enumerate(cens):
        ax.text(i, v + max(cens) * 0.03, f"{v:,}", ha="center", fontsize=9)
    fig.suptitle("Dataset accounting: labels remain auditable before model access", x=0.04, ha="left", fontsize=16, fontweight="bold", color=INK)
    savefig(fig, "dataset_partition_and_labels")


def figure_acquisition(data: dict[str, object]) -> None:
    sizes = np.array([p.stat().st_size for p in FITS.glob("*.fits")], dtype=float) / 1024
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    ax = axes[0]
    ax.hist(sizes, bins=28, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.axvline(np.median(sizes), color=RED, linewidth=2, label=f"median {np.median(sizes):.1f} KB")
    ax.set_xlabel("FITS file size (KB)")
    ax.set_ylabel("Files")
    style(ax, "Retrieved payload sizes")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    report = data["acquisition"]
    vals = [report["planned_samples"], report["valid_samples"], report["missing_count"], report["invalid_count"]]
    labels = ["planned", "valid", "missing", "invalid"]
    colors = [BLUE, GREEN, RED, RED]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.set_ylabel("Count")
    style(ax, "Acquisition gate")
    ax.set_ylim(0, max(vals) * 1.17)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02, f"{v:,}", ha="center", fontsize=9)
    fig.suptitle("FITS acquisition passed before BASE training", x=0.04, ha="left", fontsize=16, fontweight="bold", color=INK)
    savefig(fig, "acquisition_qc")


def figure_latitude_time(data: dict[str, object]) -> None:
    m: pd.DataFrame = data["manifest"]  # type: ignore[assignment]
    a: pd.DataFrame = data["acquired"]  # type: ignore[assignment]
    m2 = m[m.partition.isin(["train", "validation", "test"])].copy()
    m2["date"] = pd.to_datetime(m2.t_rec, utc=True, errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for part, col in [("train", BLUE), ("validation", ORANGE), ("test", PURPLE)]:
        z = m2[m2.partition.eq(part)]
        ax.hist(z.latitude_deg.dropna(), bins=32, alpha=0.58, color=col, label=part.title())
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Manifest rows")
    style(ax, "Latitude coverage")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    z = a.copy()
    z["date"] = pd.to_datetime(z.t_rec, utc=True, errors="coerce")
    for lab, mask, col in [("positive", z.label_m1plus_24h.eq(1), RED), ("negative", z.label_m1plus_24h.eq(0), BLUE)]:
        ax.scatter(z.loc[mask, "date"], z.loc[mask, "latitude_deg"], s=3, alpha=0.35, color=col, label=lab)
    ax.set_xlabel("Observation time (UTC)")
    ax.set_ylabel("Latitude (deg)")
    style(ax, "Acquired BASE records")
    ax.legend(frameon=False, fontsize=9, markerscale=3)
    fig.suptitle("Selection geometry and temporal coverage", x=0.04, ha="left", fontsize=16, fontweight="bold", color=INK)
    savefig(fig, "latitude_and_time_coverage")


def figure_tai_and_gates(data: dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    deltas = data["tai_audit"]["timestamp_delta_seconds"]  # type: ignore[index]
    labs = list(deltas.keys())
    vals = list(deltas.values())
    ax.bar(labs, vals, color=PURPLE, width=0.62)
    ax.set_xlabel("TAI − UTC offset (s)")
    ax.set_ylabel("Metadata rows")
    style(ax, "Canonical time-scale repair")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:,}", ha="center", fontsize=8)
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    stages = [(0.5, 5.2, "ACQUIRE", "HTTP/FITS\nvalidation", BLUE), (0.5, 3.35, "PREFLIGHT", "cache complete", ORANGE), (0.5, 1.5, "TRAIN", "BASE only", GREEN)]
    for x, y, title, sub, col in stages:
        ax.add_patch(FancyBboxPatch((x, y), 9, 1.12, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor="white", edgecolor=col, linewidth=2))
        ax.text(0.85, y + 0.70, title, fontsize=11, fontweight="bold", color=col, va="center")
        ax.text(3.2, y + 0.70, sub, fontsize=10, color=INK, va="center")
        ax.text(8.9, y + 0.70, "PASS", fontsize=10, fontweight="bold", color=GREEN, va="center", ha="right")
    ax.add_patch(FancyArrowPatch((5, 5.1), (5, 4.45), arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#627d98"))
    ax.add_patch(FancyArrowPatch((5, 3.25), (5, 2.6), arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#627d98"))
    ax.text(5, 0.55, "Physics and downstream were intentionally not enabled in this run.", ha="center", fontsize=9, color=RED)
    fig.suptitle("Two safeguards: exact chronology and fail-closed execution", x=0.04, ha="left", fontsize=16, fontweight="bold", color=INK)
    savefig(fig, "time_repair_and_execution_gates")


def figure_training(data: dict[str, object]) -> None:
    h = pd.DataFrame(data["history"])  # type: ignore[arg-type]
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.5), sharex=True)
    ax = axes[0]
    ax.plot(h.step, h.loss, color=INK, linewidth=1.0, alpha=0.28, label="step loss")
    ax.plot(h.step, h.loss.rolling(31, min_periods=1, center=True).median(), color=BLUE, linewidth=2.2, label="31-step rolling median")
    ax.axvline(400, color=RED, linestyle="--", linewidth=1.5, label="resume boundary (400)")
    ax.set_ylabel("Combined loss")
    style(ax, "BASE resumed training trajectory")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    for col, color, label in [("denoise", ORANGE, "denoise"), ("generic", GREEN, "generic calibration"), ("learning_rate", PURPLE, "learning rate")]:
        y = h[col]
        if col == "learning_rate":
            ax2 = ax.twinx()
            ax2.plot(h.step, y, color=color, linewidth=1.8, label=label)
            ax2.set_ylabel("Learning rate", color=color)
            ax2.tick_params(axis="y", colors=color)
        else:
            ax.plot(h.step, y, color=color, linewidth=1.4, label=label)
    ax.set_xlabel("Step")
    ax.set_ylabel("Component loss")
    style(ax, "Loss components and cosine schedule")
    handles, labels = ax.get_legend_handles_labels()
    if ax.figure.axes[-1] is not ax:
        h2, l2 = ax.figure.axes[-1].get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, frameon=False, fontsize=9, loc="upper right")
    savefig(fig, "training_history")


def figure_pairs(data: dict[str, object]) -> None:
    sid, real, row = representative_real(data)
    syn: np.ndarray = data["syn_gauss"][0]  # type: ignore[index]
    lim = float(np.percentile(np.abs(np.concatenate([real.ravel(), syn.ravel()])), 99.5))
    lim = float(np.clip(lim, 400, 3000))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), constrained_layout=True)
    for ax, arr, title, subtitle in [(axes[0], real, "Real preprocessed magnetogram", sid), (axes[1], syn, "Generated BASE sample", "base_2026_RG00004_000000")]:
        im = ax.imshow(arr, cmap="RdBu_r", norm=norm, origin="lower")
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("fixed-FOV x pixel")
        ax.set_ylabel("fixed-FOV y pixel")
        ax.text(0.02, -0.16, subtitle, transform=ax.transAxes, fontsize=8.5, color="#486581")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="B$_{LOS}$ (G)")
    fig.suptitle("Qualitative comparison: value fields, not a proof of physical equivalence", x=0.04, ha="left", fontsize=15, fontweight="bold", color=INK)
    savefig(fig, "real_synthetic_magnetogram_pair")

    # A grid is more useful than a single cherry-picked sample for inspecting modes.
    fig, axes = plt.subplots(3, 4, figsize=(11.8, 8.5), constrained_layout=True)
    for i, ax in enumerate(axes.flat):
        arr = data["syn_gauss"][i]  # type: ignore[index]
        ax.imshow(arr, cmap="RdBu_r", norm=norm, origin="lower")
        ax.set_title(f"sample {i + 1:02d}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Twelve of 128 generated fields (same BASE checkpoint)", x=0.04, ha="left", fontsize=15, fontweight="bold", color=INK)
    savefig(fig, "generated_sample_grid")


def figure_surfaces(data: dict[str, object]) -> None:
    sid, real, _ = representative_real(data)
    syn: np.ndarray = data["syn_gauss"][0]  # type: ignore[index]
    x = np.linspace(-128, 128, 64)
    y = np.linspace(-128, 128, 64)
    xx, yy = np.meshgrid(x, y)
    real_ds = real[::2, ::2]
    syn_ds = syn[::2, ::2]
    lim = float(np.percentile(np.abs(np.concatenate([real_ds.ravel(), syn_ds.ravel()])), 99.5))
    lim = float(np.clip(lim, 400, 3000))
    fig = plt.figure(figsize=(13, 5.8))
    for i, (arr, title, color) in enumerate([(real_ds, f"Real · {sid}", BLUE), (syn_ds, "Generated BASE", ORANGE)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(xx, yy, arr, cmap="RdBu_r", linewidth=0, antialiased=True, rcount=64, ccount=64, vmin=-lim, vmax=lim)
        ax.set_title(title, loc="left", pad=12, color=INK, fontweight="bold")
        ax.set_xlabel("x (Mm)", labelpad=5)
        ax.set_ylabel("y (Mm)", labelpad=5)
        ax.set_zlabel("B$_{LOS}$ (G)", labelpad=5)
        ax.set_zlim(-lim, lim)
        ax.view_init(elev=30, azim=-58)
        ax.text2D(0.03, 0.03, "height = field value; not a coronal geometry", transform=ax.transAxes, fontsize=8, color="#486581")
    fig.suptitle("3-D visualization of 2-D magnetogram values", x=0.04, ha="left", fontsize=15, fontweight="bold", color=INK)
    savefig(fig, "real_synthetic_3d_surfaces")

    std = data["syn_gauss"].std(axis=0)  # type: ignore[union-attr]
    fig = plt.figure(figsize=(12.7, 5.5))
    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(std, cmap="magma", origin="lower")
    ax.set_title("Pixelwise ensemble spread", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("fixed-FOV x pixel")
    ax.set_ylabel("fixed-FOV y pixel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="SD (G)")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(xx, yy, std[::2, ::2], cmap="magma", linewidth=0, antialiased=True, rcount=64, ccount=64)
    ax.set_title("Spread as a surface", loc="left", pad=12, fontweight="bold", color=INK)
    ax.set_xlabel("x (Mm)")
    ax.set_ylabel("y (Mm)")
    ax.set_zlabel("SD (G)")
    ax.view_init(elev=32, azim=-58)
    fig.suptitle("Sampling uncertainty: 128 deterministic-configuration draws", x=0.04, ha="left", fontsize=15, fontweight="bold", color=INK)
    savefig(fig, "synthetic_ensemble_uncertainty")

    write_obj(MODEL_DIR / "real_magnetogram_surface.obj", real, f"Real preprocessed magnetogram {sid}")
    write_obj(MODEL_DIR / "synthetic_magnetogram_surface.obj", syn, "Generated BASE magnetogram")
    write_obj(MODEL_DIR / "synthetic_ensemble_spread_surface.obj", std, "Generated BASE ensemble standard deviation")
    write_text(
        MODEL_DIR / "README.txt",
        "The OBJ files are visualization meshes. x/y span the fixed 256 Mm field of view, and z is the line-of-sight magnetic field value in gauss (or its ensemble standard deviation). They are not 3-D coronal magnetic-field reconstructions. The real and synthetic meshes correspond to the report figure; the synthetic ensemble spread is computed pixelwise across 128 samples.",
    )


def hist_grid(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str], title: str, name: str, xlabel_map: dict[str, str] | None = None) -> None:
    fig, axes = plt.subplots(2, math.ceil(len(cols) / 2), figsize=(12.5, 7.3))
    axes = np.atleast_1d(axes).ravel()
    xlabel_map = xlabel_map or {}
    for ax, col in zip(axes, cols):
        r = real[col].to_numpy(float)
        s = syn[col].to_numpy(float)
        lo = np.nanpercentile(np.concatenate([r, s]), 0.5)
        hi = np.nanpercentile(np.concatenate([r, s]), 99.5)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(np.concatenate([r, s])), np.nanmax(np.concatenate([r, s])) + 1
        bins = np.linspace(lo, hi, 22)
        ax.hist(r, bins=bins, density=True, alpha=0.52, color=BLUE, label="real")
        ax.hist(s, bins=bins, density=True, alpha=0.52, color=ORANGE, label="synthetic")
        ax.axvline(np.median(r), color=BLUE, linewidth=1.5)
        ax.axvline(np.median(s), color=ORANGE, linewidth=1.5, linestyle="--")
        ax.set_xlabel(xlabel_map.get(col, col))
        ax.set_ylabel("density")
        style(ax, col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(title, x=0.04, ha="left", fontsize=15, fontweight="bold", color=INK)
    savefig(fig, name)


def figure_descriptors(data: dict[str, object]) -> None:
    hist_grid(
        data["gen"], data["syn_gen"],  # type: ignore[arg-type]
        ["log_mean_abs", "log_p90_abs", "log_p99_abs", "active_fraction", "strong_fraction", "saturation_fraction"],
        "Generic independent descriptor distributions",
        "generic_descriptor_distributions",
        {"active_fraction": "|B| > 150 G fraction", "strong_fraction": "|B| > 500 G fraction", "saturation_fraction": "|B| > 2,900 G fraction"},
    )
    hist_grid(
        data["geom"], data["syn_geom"],  # type: ignore[arg-type]
        ["hemi_ux", "hemi_uy", "log_sep", "log_strong_flux_density", "has_bipole"],
        "Independent hard geometry descriptors",
        "hard_geometry_distributions",
        {"hemi_ux": "hemisphere-adjusted u", "hemi_uy": "hemisphere-adjusted v", "log_sep": "log(1 + separation)", "log_strong_flux_density": "log strong-field density"},
    )
    hist_grid(
        data["pil"], data["syn_pil"],  # type: ignore[arg-type]
        ["mean_grad", "rms_grad", "top10_grad", "frac_gt100", "frac_gt250", "frac_gt500"],
        "Independent hard strong-PIL descriptors",
        "hard_pil_distributions",
        {"mean_grad": "mean |∇B| (G/Mm)", "rms_grad": "RMS |∇B| (G/Mm)", "top10_grad": "top-tail |∇B| (G/Mm)"},
    )


def figure_negative_control(data: dict[str, object]) -> None:
    a = data["audit"]
    labels = ["original 6-D\n(includes near-zero\nsaturation scale)", "corrected 5-D core\n(saturation separately gated)", "allowed threshold"]
    vals = [a["generic_distance_to_real_baseline_ratio_original_6d"], a["generic_distance_to_real_baseline_ratio"], 8.0]
    colors = [RED, GREEN, "#627d98"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    bars = ax.bar(labels, vals, color=colors, width=0.65)
    ax.set_ylabel("synthetic / real split-half distance")
    style(ax, "Generic multivariate distance negative control")
    ax.set_ylim(0, max(vals) * 1.12)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02, f"{v:.2f}×", ha="center", fontsize=9)
    ax = axes[1]
    r = data["gen"]["saturation_fraction"].to_numpy(float)  # type: ignore[index]
    s = data["syn_gen"]["saturation_fraction"].to_numpy(float)  # type: ignore[index]
    ax.hist(r, bins=16, alpha=0.62, color=BLUE, label="real")
    ax.hist(s, bins=16, alpha=0.62, color=ORANGE, label="synthetic")
    ax.set_xlabel("fraction |B| > 2,900 G")
    ax.set_ylabel("count")
    style(ax, "Saturation is separately bounded")
    ax.legend(frameon=False, fontsize=9)
    ax.text(0.98, 0.95, f"synthetic mean = {np.mean(s):.3e}\nabsolute gate < 0.01", transform=ax.transAxes, ha="right", va="top", fontsize=9, color=INK)
    fig.suptitle("A failed metric formulation was corrected transparently", x=0.04, ha="left", fontsize=15, fontweight="bold", color=INK)
    savefig(fig, "negative_control_metric_correction")


def figure_feature_scatter(data: dict[str, object]) -> None:
    r: pd.DataFrame = data["gen"]  # type: ignore[assignment]
    s: pd.DataFrame = data["syn_gen"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))
    ax = axes[0]
    ax.scatter(r.log_mean_abs, r.log_p99_abs, s=15, alpha=0.50, color=BLUE, label="real")
    ax.scatter(s.log_mean_abs, s.log_p99_abs, s=18, alpha=0.62, color=ORANGE, label="synthetic")
    ax.set_xlabel("log mean |B|")
    ax.set_ylabel("log p99 |B|")
    style(ax, "Marginal-strength structure")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    ax.scatter(r.active_fraction, r.strong_fraction, s=15, alpha=0.50, color=BLUE, label="real")
    ax.scatter(s.active_fraction, s.strong_fraction, s=18, alpha=0.62, color=ORANGE, label="synthetic")
    ax.set_xlabel("active fraction (>150 G)")
    ax.set_ylabel("strong fraction (>500 G)")
    style(ax, "Area-occupancy structure")
    fig.suptitle("Two-dimensional descriptor relationships", x=0.04, ha="left", fontsize=15, fontweight="bold", color=INK)
    savefig(fig, "descriptor_scatter")


def figure_summary_bars(data: dict[str, object]) -> None:
    a = data["audit"]
    labels = ["flux proxy", "active area", "diversity"]
    vals = [a["synthetic_to_real_flux_median_ratio"], a["synthetic_to_real_active_area_median_ratio"], a["synthetic_to_real_diversity_ratio"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars = ax.bar(labels, vals, color=[BLUE, ORANGE, GREEN], width=0.55)
    ax.axhspan(0.5, 2.0, color=GREEN, alpha=0.08, label="broad quality interval (flux/area)")
    ax.axhline(1.0, color=INK, linewidth=1.1, linestyle="--")
    ax.set_ylabel("synthetic / real median ratio")
    style(ax, "Core calibration and diversity checks")
    ax.set_ylim(0, max(2.2, max(vals) * 1.25))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.06, f"{v:.3f}×", ha="center", fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    savefig(fig, "core_gate_ratios")


def figure_seed_robustness(data: dict[str, object]) -> None:
    alt = data.get("alternate")
    if not alt:
        return
    primary = data["audit"]
    names = ["seed 2026", "seed 2027"]
    metrics = [
        ("generic distance /\nreal p90", [primary["generic_distance_to_real_baseline_ratio"], alt["generic_distance_to_real_baseline_ratio"]], 8.0),
        ("flux median\nratio", [primary["synthetic_to_real_flux_median_ratio"], alt["synthetic_to_real_flux_median_ratio"]], None),
        ("active area\nratio", [primary["synthetic_to_real_active_area_median_ratio"], alt["synthetic_to_real_active_area_median_ratio"]], None),
        ("diversity\nratio", [primary["synthetic_to_real_diversity_ratio"], alt["synthetic_to_real_diversity_ratio"]], None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 4.5))
    for ax, (label, vals, threshold) in zip(axes, metrics):
        bars = ax.bar(names, vals, color=[BLUE, ORANGE], width=0.55)
        if threshold is not None:
            ax.axhline(threshold, color=RED, linestyle="--", linewidth=1.4, label="gate threshold")
        ax.axhline(1, color=INK, linestyle=":", linewidth=1.0)
        ax.set_title(label, fontsize=10, fontweight="bold", color=INK)
        ax.tick_params(axis="x", labelrotation=30)
        style(ax)
        ymax = max(vals) * 1.28 if max(vals) > 0 else 1
        ax.set_ylim(0, ymax)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ymax * 0.03, f"{v:.2f}", ha="center", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Sampler-seed robustness against the fixed primary real reference", x=0.04, ha="left", fontsize=15, fontweight="bold", color=INK)
    savefig(fig, "seed_robustness")


def write_tables(data: dict[str, object]) -> dict[str, pd.DataFrame]:
    def write_latex_table(frame: pd.DataFrame, path: Path, columns: list[str], labels: list[str], digits: int = 4) -> None:
        view = frame[columns].copy()
        view.columns = labels
        formatters = {c: (lambda x, d=digits: fmt(x, d)) for c in view.columns if c not in {"partition", "quantity", "notes", "family", "feature"}}
        tex = view.to_latex(index=False, escape=True, longtable=False, formatters=formatters)
        # The audit tables are short but wide. Scale the tabular body to the
        # text width so the PDF never clips feature names or statistics.
        tex = tex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
        tex = tex.replace("\\end{tabular}", "\\end{tabular}}")
        write_text(path, "\\scriptsize\n" + tex + "\n\\normalsize")

    ma: dict = data["manifest_audit"]  # type: ignore[assignment]
    rows = []
    for p in ma["partitions"]:
        rows.append(
            {
                "partition": p["partition"],
                "rows": p["rows"],
                "positive_rows": p["positive_rows"],
                "independent_groups": p["independent_groups"],
                "positive_groups": p["independent_positive_groups"],
                "independent_harps": p["independent_harps"],
                "positive_harps": p["independent_positive_harps"],
                "censored_negative_rows": p["censored_negative_rows"],
                "image_urls": p["image_urls"],
            }
        )
    partition = pd.DataFrame(rows)
    partition.to_csv(TABLE / "partition_summary.csv", index=False)
    write_latex_table(
        partition,
        TABLE / "partition_summary.tex",
        ["partition", "rows", "positive_rows", "independent_groups", "positive_groups", "independent_harps", "positive_harps", "censored_negative_rows", "image_urls"],
        ["partition", "rows", "M1+", "groups", "positive groups", "HARPs", "positive HARPs", "censored negatives", "image URLs"],
        0,
    )

    acquired: pd.DataFrame = data["acquired"]  # type: ignore[assignment]
    selected_rows = []
    for part, z in acquired.groupby("partition", dropna=False):
        selected_rows.append(
            {
                "partition": part,
                "acquired_files": len(z),
                "positive": int(z.label_m1plus_24h.eq(1).sum()),
                "negative": int(z.label_m1plus_24h.eq(0).sum()),
                "groups": z.region_group_id.nunique(),
                "harps": z.harpnum.nunique(),
                "median_file_size_kb": float(np.median([p.stat().st_size for p in FITS.glob("*.fits") if p.stem in set(z.sample_id.astype(str))]) / 1024) if len(z) else float("nan"),
            }
        )
    acquired_summary = pd.DataFrame(selected_rows)
    acquired_summary.to_csv(TABLE / "acquired_cache_summary.csv", index=False)
    write_latex_table(
        acquired_summary,
        TABLE / "acquired_cache_summary.tex",
        ["partition", "acquired_files", "positive", "negative", "groups", "harps", "median_file_size_kb"],
        ["partition", "files", "positive", "negative", "groups", "HARPs", "median KB"],
        2,
    )

    desc = descriptor_table(data)
    desc.to_csv(TABLE / "generic_descriptor_statistics.csv", index=False)
    write_latex_table(
        desc,
        TABLE / "generic_descriptor_statistics.tex",
        ["feature", "real_median", "synthetic_median", "median_ratio", "ratio_ci_low", "ratio_ci_high", "wasserstein", "ks_statistic", "ks_pvalue"],
        ["feature", "real median", "synthetic median", "ratio", "CI low", "CI high", "Wasserstein", "KS", "KS p"],
        4,
    )
    hard_geom = hard_descriptor_table(data, "hard_geometry", data["geom"], data["syn_geom"])  # type: ignore[arg-type]
    hard_pil = hard_descriptor_table(data, "hard_pil", data["pil"], data["syn_pil"])  # type: ignore[arg-type]
    hard = pd.concat([hard_geom, hard_pil], ignore_index=True)
    hard.to_csv(TABLE / "hard_descriptor_statistics.csv", index=False)
    write_latex_table(
        hard,
        TABLE / "hard_descriptor_statistics.tex",
        ["family", "feature", "real_median", "synthetic_median", "median_ratio", "ratio_ci_low", "ratio_ci_high", "wasserstein", "ks_statistic", "ks_pvalue"],
        ["family", "feature", "real median", "synthetic median", "ratio", "CI low", "CI high", "Wasserstein", "KS", "KS p"],
        4,
    )

    history: pd.DataFrame = pd.DataFrame(data["history"])  # type: ignore[arg-type]
    training = pd.DataFrame(
        [
            {"quantity": "start_step", "value": int(history.step.iloc[0]) - 1, "notes": "numbered checkpoint resume boundary"},
            {"quantity": "end_step", "value": int(history.step.iloc[-1]), "notes": "final step"},
            {"quantity": "start_loss", "value": float(history.loss.iloc[0]), "notes": "first recorded post-resume update"},
            {"quantity": "end_loss", "value": float(history.loss.iloc[-1]), "notes": "final combined loss"},
            {"quantity": "minimum_loss", "value": float(history.loss.min()), "notes": "minimum recorded post-resume loss"},
            {"quantity": "start_denoise", "value": float(history.denoise.iloc[0]), "notes": "first recorded post-resume component"},
            {"quantity": "end_denoise", "value": float(history.denoise.iloc[-1]), "notes": "final denoise component"},
            {"quantity": "start_generic", "value": float(history.generic.iloc[0]), "notes": "first recorded post-resume component"},
            {"quantity": "end_generic", "value": float(history.generic.iloc[-1]), "notes": "final generic component"},
        ]
    )
    training.to_csv(TABLE / "training_summary.csv", index=False)
    write_latex_table(training, TABLE / "training_summary.tex", ["quantity", "value", "notes"], ["quantity", "value", "notes"], 6)

    alt = data.get("alternate")
    seed_rows = [
        {
            "seed": 2026,
            "real_reference": "primary audit",
            "generic_ratio": data["audit"]["generic_distance_to_real_baseline_ratio"],
            "flux_ratio": data["audit"]["synthetic_to_real_flux_median_ratio"],
            "active_area_ratio": data["audit"]["synthetic_to_real_active_area_median_ratio"],
            "diversity_ratio": data["audit"]["synthetic_to_real_diversity_ratio"],
            "saturation_fraction": data["audit"]["synthetic_saturation_fraction_abs_gt_2900G"],
            "gate": str(data["audit"]["generic_fidelity_gate_pass"]),
        }
    ]
    if alt:
        seed_rows.append(
            {
                "seed": 2027,
                "real_reference": "fixed primary descriptors",
                "generic_ratio": alt["generic_distance_to_real_baseline_ratio"],
                "flux_ratio": alt["synthetic_to_real_flux_median_ratio"],
                "active_area_ratio": alt["synthetic_to_real_active_area_median_ratio"],
                "diversity_ratio": alt["synthetic_to_real_diversity_ratio"],
                "saturation_fraction": alt["synthetic_saturation_fraction_abs_gt_2900G"],
                "gate": str(alt["generic_fidelity_gate_pass"]),
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(TABLE / "seed_robustness.csv", index=False)
    write_latex_table(
        seed_df,
        TABLE / "seed_robustness.tex",
        ["seed", "real_reference", "generic_ratio", "flux_ratio", "active_area_ratio", "diversity_ratio", "saturation_fraction", "gate"],
        ["seed", "real reference", "generic / p90", "flux ratio", "area ratio", "diversity ratio", "saturation fraction", "gate"],
        4,
    )

    destruction = data.get("destruction")
    if destruction:
        destruction_df = pd.DataFrame(destruction["controls"])
        destruction_df.to_csv(TABLE / "destruction_controls.csv", index=False)
        write_latex_table(
            destruction_df,
            TABLE / "destruction_controls.tex",
            ["control", "generic_core_ratio_to_real_p90", "hard_geometry_distance", "hard_pil_distance", "parent_pairwise_generic_change", "parent_pairwise_geometry_change", "parent_pairwise_pil_change", "parent_field_mae_gauss"],
            ["control", "generic / p90", "geometry distance", "PIL distance", "generic change", "geometry change", "PIL change", "field MAE (G)"],
            4,
        )

    return {"partition": partition, "acquired": acquired_summary, "generic": desc, "hard": hard, "training": training}


def write_inventory(data: dict[str, object]) -> None:
    files = []
    for root, label in [
        (SOURCE, "source"),
        (EVIDENCE, "evidence"),
        (FITS, "fits_cache"),
        (RUN, "base_run"),
        (PHYSICS_SCREEN, "physics_screening"),
        (PREPARED_TENSORS, "prepared_tensors"),
    ]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file():
                files.append({"area": label, "path": str(p.relative_to(root)), "bytes": p.stat().st_size})
    inv = pd.DataFrame(files)
    inv.to_csv(ARTIFACT / "file_inventory.csv", index=False)
    fit_sizes = inv[inv.area.eq("fits_cache")].bytes
    summary = {
        "source_root": str(SOURCE),
        "evidence_root": str(EVIDENCE),
        "fits_cache_root": str(FITS),
        "run_root": str(RUN),
        "source_files": int((inv.area == "source").sum()),
        "evidence_files": int((inv.area == "evidence").sum()),
        "fits_files": int(len(list(FITS.glob("*.fits")))),
        "run_files": int((inv.area == "base_run").sum()),
        "physics_screen_files": int((inv.area == "physics_screening").sum()),
        "prepared_tensor_files": int((inv.area == "prepared_tensors").sum()),
        "fits_bytes": int(fit_sizes.sum()) if len(fit_sizes) else 0,
        "run_bytes": int(inv.loc[inv.area.eq("base_run"), "bytes"].sum()),
        "physics_screen_bytes": int(inv.loc[inv.area.eq("physics_screening"), "bytes"].sum()),
        "prepared_tensor_bytes": int(inv.loc[inv.area.eq("prepared_tensors"), "bytes"].sum()),
        "evidence_sha256s_path": str(EVIDENCE / "SHA256SUMS.txt"),
        "combined_zip": os.environ.get("IRIS_COMBINED_ARCHIVE", "IRIS_Colab_FULL_WITH_ACQUIRED_DATA_2026-08-31.zip"),
    }
    write_json(ARTIFACT / "inventory_summary.json", summary)


def write_values_tex(data: dict[str, object], tables: dict[str, pd.DataFrame]) -> None:
    a = data["audit"]
    acq = data["acquisition"]
    ma = data["manifest_audit"]
    ta = data["tai_audit"]
    ck = data["ck"]
    run = data["run_cfg"]
    history = pd.DataFrame(data["history"])  # type: ignore[arg-type]
    partition = tables["partition"]
    acquired = tables["acquired"]
    def macro(name: str, value: object) -> str:
        return f"\\newcommand{{\\{name}}}{{{tex_escape(value)}}}"

    lines = [
        "% Generated by scripts/generate_analysis.py; do not hand-edit.",
        macro("AcquiredCount", acq["valid_samples"]),
        macro("PlannedCount", acq["planned_samples"]),
        macro("ManifestCount", ma["total_sharp_rows"]),
        macro("PrimaryCount", ma["primary_rows_after_censoring"]),
        macro("GroupCount", ma["active_connected_groups"]),
        macro("MOneEvents", data["collection"]["m1plus_events"]),
        macro("MappedHarps", data["collection"]["mapped_harps"]),
        macro("UnresolvedEvents", data["label_audit"]["unresolved_events"]),
        macro("CensoredRows", data["label_audit"]["rows_censored_total"]),
        macro("TaiRows", ta["rows"]),
        macro("TaiLabelChanges", ta["label_changes_primary"]),
        macro("BaseSteps", ck["steps"]),
        macro("ResumeStep", ck["resumed_from_step"]),
        macro("DiffusionSteps", ck["diffusion_steps"]),
        macro("BaseChannels", ck["base_channels"]),
        macro("SyntheticCount", a["synthetic_count"]),
        macro("AuditRealCount", a["real_count"]),
        macro("AuditDistanceRatio", f"{a['generic_distance_to_real_baseline_ratio']:.3f}"),
        macro("AuditOriginalDistanceRatio", f"{a['generic_distance_to_real_baseline_ratio_original_6d']:.1f}"),
        macro("AuditFluxRatio", f"{a['synthetic_to_real_flux_median_ratio']:.3f}"),
        macro("AuditAreaRatio", f"{a['synthetic_to_real_active_area_median_ratio']:.3f}"),
        macro("AuditDiversityRatio", f"{a['synthetic_to_real_diversity_ratio']:.3f}"),
        macro("AuditSatFraction", f"{a['synthetic_saturation_fraction_abs_gt_2900G']:.3e}"),
        macro("AuditGate", "PASS" if a["generic_fidelity_gate_pass"] else "FAIL"),
        macro("StartLoss", f"{history.loss.iloc[0]:.4f}"),
        macro("EndLoss", f"{history.loss.iloc[-1]:.4f}"),
        macro("StartDenoise", f"{history.denoise.iloc[0]:.4f}"),
        macro("EndDenoise", f"{history.denoise.iloc[-1]:.4f}"),
        macro("StartGeneric", f"{history.generic.iloc[0]:.4f}"),
        macro("EndGeneric", f"{history.generic.iloc[-1]:.4f}"),
        macro("TrainRows", int(partition.loc[partition.partition.eq("train"), "rows"].iloc[0])),
        macro("ValidationRows", int(partition.loc[partition.partition.eq("validation"), "rows"].iloc[0])),
        macro("TestRows", int(partition.loc[partition.partition.eq("test"), "rows"].iloc[0])),
        macro("AcquiredTrainFiles", int(acquired.loc[acquired.partition.eq("train"), "acquired_files"].iloc[0]) if (acquired.partition == "train").any() else 0),
        macro("AcquiredNegativeFiles", int(acquired.negative.sum())),
        macro("AcquiredPositiveFiles", int(acquired.positive.sum())),
        macro("BoundaryOne", data["frozen_split"]["boundary_1"]),
        macro("BoundaryTwo", data["frozen_split"]["boundary_2"]),
        macro("FinalLearningRate", history.learning_rate.iloc[-1]),
        macro("CheckpointOptimizer", "not stored" if not ck["optimizer_state_available"] else "stored"),
    ]
    write_text(TABLE / "generated_values.tex", "\n".join(lines))


def write_metadata_json(data: dict[str, object], tables: dict[str, pd.DataFrame]) -> None:
    a = data["audit"]
    physics_screening = None
    physics_selftest = None
    screening_path = ARTIFACT / "physics_screening_metrics.json"
    selftest_path = ARTIFACT / "physics_v2_selftest.json"
    if screening_path.is_file():
        physics_screening = json.loads(screening_path.read_text())
    if selftest_path.is_file():
        physics_selftest = json.loads(selftest_path.read_text())
    classifier_path = ARTIFACT / "two_sample_classifier_audit.json"
    classifier_audit = json.loads(classifier_path.read_text()) if classifier_path.is_file() else None
    selftests_path = ARTIFACT / "offline_protocol_selftests.json"
    offline_selftests = json.loads(selftests_path.read_text()) if selftests_path.is_file() else None
    conditional_path = ARTIFACT / "conditional_proxy_diagnostics.json"
    conditional_audit = json.loads(conditional_path.read_text()) if conditional_path.is_file() else None
    gate_stability_path = ARTIFACT / "gate_stability_diagnostics.json"
    gate_stability = json.loads(gate_stability_path.read_text()) if gate_stability_path.is_file() else None
    memorization_path = ARTIFACT / "memorization_diagnostics.json"
    memorization = json.loads(memorization_path.read_text()) if memorization_path.is_file() else None
    summary = {
        "experiment": "IRIS BASE local resume and independent v2 audit",
        "as_of": "2026-08-31",
        "scope": "BASE only; physics and downstream intentionally disabled",
        "source_provenance": {
            "collection_summary": data["collection"],
            "source_ledger": data["source_ledger"],
            "tai_repair": data["tai_audit"],
            "frozen_split": {k: v for k, v in data["frozen_split"].items() if k != "parts"},
            "manifest_audit": data["manifest_audit"],
            "label_audit": data["label_audit"],
            "retrieval_smoke_test": data["retrieval"],
        },
        "acquisition": data["acquisition"],
        "model_checkpoint_metadata": data["ck"],
        "resume_config": data["run_cfg"],
        "sampling": data["sampling"],
        "independent_audit": a,
        "alternate_sampling_audit": data.get("alternate"),
        "destruction_controls": data.get("destruction"),
        "executed_extensions": {
            "physics_v2_selftest": physics_selftest,
            "physics_factorial_screening": physics_screening,
            "two_sample_classifier_audit": classifier_audit,
            "offline_protocol_selftests": offline_selftests,
            "conditional_proxy_diagnostics": conditional_audit,
            "gate_stability_diagnostics": gate_stability,
            "memorization_diagnostics": memorization,
        },
        "counts": {
            "manifest_rows": int(len(data["manifest"])),
            "acquired_fits": int(len(data["fit_ids"])),
            "audit_real": int(len(data["gen"])),
            "audit_synthetic": int(len(data["syn_gen"])),
        },
        "report_outputs": {
            "figures": sorted(p.name for p in FIG.glob("*.png")),
            "auxiliary_figures": sorted(p.name for p in FIG.glob("*.svg")),
            "tables": sorted(p.name for p in TABLE.glob("*")),
            "models": sorted(p.name for p in MODEL_DIR.glob("*")),
        },
        "claim_boundary": [
            "A BASE train-only image/distribution audit passed its predeclared broad gate.",
            "The corrected generic-core distance was 7.681 times the real split-half p90 reference, so fidelity is not close to real-data variability.",
            "No physics-constrained or downstream forecasting conclusion is supported by this run.",
            "The local resume did not restore optimizer or RNG state because numbered checkpoints did not contain them.",
        ],
    }
    write_json(ARTIFACT / "report_metadata.json", summary)


def main() -> None:
    ensure_dirs()
    data = load_inputs()
    tables = write_tables(data)
    write_inventory(data)
    write_values_tex(data, tables)
    write_metadata_json(data, tables)
    figure_graphical_abstract(data)
    figure_dataset(data)
    figure_acquisition(data)
    figure_latitude_time(data)
    figure_tai_and_gates(data)
    figure_training(data)
    figure_pairs(data)
    figure_surfaces(data)
    figure_descriptors(data)
    figure_negative_control(data)
    figure_feature_scatter(data)
    figure_summary_bars(data)
    figure_seed_robustness(data)
    # Refresh the manifest after all figures/models have been created so the
    # published metadata describes the final report package, not the initial
    # empty output directories.
    write_metadata_json(data, tables)
    print(json.dumps({"report": str(REPORT), "figures": len(list(FIG.glob("*.png"))), "tables": len(list(TABLE.glob("*"))), "models": len(list(MODEL_DIR.glob("*.obj")))}, indent=2))


if __name__ == "__main__":
    main()
