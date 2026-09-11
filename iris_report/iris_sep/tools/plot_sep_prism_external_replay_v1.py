#!/usr/bin/env python3
"""Generate publication figures from the frozen SEP-PRISM replay tables."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = ROOT / "replay"
OUT = Path(__file__).resolve().parents[1] / "figures"
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#6B7280"


def save_all(fig: plt.Figure, stem: str) -> None:
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)


def tss_figure() -> None:
    points = pd.read_csv(ARTIFACT / "matched_episode_point_results.csv")
    order = ["MAPPED_STANDARD", "EPISODE_NORMALIZED_OCCURRENCE", "NEW_ONSET"]
    labels = ["Mapped\noccurrence", "Episode-normalized\noccurrence", "New onset"]
    models = [
        ("xgb_joint", "Joint XGBoost", BLUE),
        ("xgb_no_proton", "No-proton XGBoost", ORANGE),
        ("past_proton_ge10_proxy", "Past-proton proxy", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for model, label, color in models:
        data = points[points.model.eq(model)].set_index("view").loc[order]
        ax.plot(labels, data.tss, marker="o", linewidth=2.6, markersize=7, label=label, color=color)
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_ylabel("True Skill Statistic (TSS)")
    fig.suptitle("Measured SEP forecast skill changes when repeated episodes and persistence are removed", x=0.09, y=0.97, ha="left", fontsize=17, weight="bold")
    fig.text(0.09, 0.91, "Matched cohort: 85 onset episodes and 1,080 quiet blocks; predictions and thresholds unchanged.", color="#374151")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.27)
    save_all(fig, "sep_prism_fixed_model_tss_2026-09-10")
    plt.close(fig)


def graphical_abstract() -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.04, "14,464 daily windows\n650 stored positives\n0 target mismatches", BLUE),
        (0.36, "614 mapped windows\n257 physical episodes\nEMF = 2.389", ORANGE),
        (0.68, "Same fixed XGBoost alerts\nTSS: 0.726 → 0.621 → 0.437\n85 matched onsets", GREEN),
    ]
    for x, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.34), 0.27, 0.32, boxstyle="round,pad=0.018", facecolor=color, alpha=0.13, edgecolor=color, linewidth=2))
        ax.text(x + 0.135, 0.50, text, ha="center", va="center", fontsize=13.5, weight="bold")
    for start in (0.315, 0.635):
        ax.annotate("", xy=(start + 0.04, 0.50), xytext=(start, 0.50), arrowprops={"arrowstyle": "->", "lw": 2.2, "color": GRAY})
    ax.text(0.04, 0.90, "One physical storm can generate several positive windows", fontsize=20, weight="bold")
    ax.text(0.04, 0.81, "Some windows begin after the storm is already active.", fontsize=15, color="#374151")
    ax.text(0.04, 0.18, "Preregistered public historical replay confirms that evaluation design materially changes measured skill.", fontsize=14)
    ax.text(0.04, 0.09, "Previously exposed public data; methodological confirmation, not prospective operational validation.", fontsize=11.5, color="#4B5563")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save_all(fig, "sep_prism_graphical_abstract_2026-09-10")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    tss_figure()
    graphical_abstract()
