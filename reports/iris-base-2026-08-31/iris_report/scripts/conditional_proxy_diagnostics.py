#!/usr/bin/env python3
"""Train-only conditional diagnostics for real versus generated proxy values.

The primary BASE audit compares pooled descriptor distributions.  This
secondary diagnostic asks whether the same conclusion survives conditioning
on the latitude bands supplied to the generator.  It uses connected-region
bootstrap units, never reads validation/test flare outcomes, and does not
alter any frozen gate.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPORT_DIR = Path(os.environ.get("IRIS_REPORT_DIR", Path(__file__).resolve().parents[1]))
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
ARTIFACT_DIR = REPORT_DIR / "artifacts"

PROXY_COLUMNS = [
    "unsigned_flux_proxy",
    "signed_flux_imbalance",
    "rms_field",
    "field_energy_proxy",
    "gradient_rms",
    "high_gradient_fraction",
    "positive_components",
    "negative_components",
    "strong_component_count",
    "strong_centroid_separation",
    "spectral_centroid",
    "spectral_high_fraction",
    "spectral_potential_proxy",
]

PROXY_LABELS = {
    "unsigned_flux_proxy": "mean |B|",
    "signed_flux_imbalance": "signed imbalance",
    "rms_field": "RMS B",
    "field_energy_proxy": "field energy",
    "gradient_rms": "gradient RMS",
    "high_gradient_fraction": "high-gradient fraction",
    "positive_components": "positive components",
    "negative_components": "negative components",
    "strong_component_count": "total components",
    "strong_centroid_separation": "centroid separation",
    "spectral_centroid": "spectral centroid",
    "spectral_high_fraction": "high-k fraction",
    "spectral_potential_proxy": "spectral potential",
}

BAND_EDGES = [-90.0, -15.0, 0.0, 15.0, 30.0, 90.0]
BAND_LABELS = ["[-90,-15]", "(-15,0]", "(0,15]", "(15,30]", "(30,90]"]
FIGURE_FEATURES = [
    "unsigned_flux_proxy",
    "signed_flux_imbalance",
    "field_energy_proxy",
    "gradient_rms",
    "high_gradient_fraction",
    "strong_component_count",
    "spectral_high_fraction",
    "spectral_potential_proxy",
]


def bootstrap_ratio(real: pd.DataFrame, synthetic: pd.DataFrame, feature: str, seed: int, reps: int) -> tuple[float, float, float]:
    real_groups = {str(group): values[feature].to_numpy(float) for group, values in real.groupby("region_group_id")}
    synthetic_groups = {str(group): values[feature].to_numpy(float) for group, values in synthetic.groupby("region_group_id")}
    real_ids = np.array(sorted(real_groups), dtype=object)
    synthetic_ids = np.array(sorted(synthetic_groups), dtype=object)
    if not len(real_ids) or not len(synthetic_ids):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=float)
    for i in range(reps):
        real_draw_ids = rng.choice(real_ids, size=len(real_ids), replace=True)
        synthetic_draw_ids = rng.choice(synthetic_ids, size=len(synthetic_ids), replace=True)
        real_draw = np.concatenate([real_groups[group] for group in real_draw_ids])
        synthetic_draw = np.concatenate([synthetic_groups[group] for group in synthetic_draw_ids])
        real_median = float(np.median(real_draw))
        synthetic_median = float(np.median(synthetic_draw))
        draws[i] = synthetic_median / max(abs(real_median), 1e-12)
    return float(np.median(draws)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_svg(rows: pd.DataFrame, path: Path) -> None:
    """Write a small dependency-free ratio heatmap for the selected proxies."""

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    def text(x: float, y: float, value: object, size: int = 11, anchor: str = "start", fill: str = "#1f2937", weight: str = "normal") -> str:
        return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}px" text-anchor="{anchor}" fill="{fill}" font-family="Arial,Helvetica,sans-serif" font-weight="{weight}">{esc(value)}</text>'

    def colour(ratio: float) -> str:
        if not np.isfinite(ratio):
            return "#e5e7eb"
        value = float(np.clip(np.log2(max(ratio, 1e-12)), -2.0, 2.0))
        if value < 0:
            strength = int(235 - 90 * abs(value) / 2.0)
            return f"rgb({strength},{strength + 12},255)"
        strength = int(235 - 90 * value / 2.0)
        return f"rgb(255,{strength + 18},{strength})"

    cell_w, cell_h = 142, 36
    left, top = 255, 115
    width, height = left + cell_w * len(BAND_LABELS) + 30, top + cell_h * len(FIGURE_FEATURES) + 95
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        text(24, 30, "Conditional proxy ratios by latitude band", 18, weight="bold"),
        text(24, 52, "synthetic median / real median; cell text is the ratio, and the footer gives real/synthetic row counts", 11, fill="#4b5563"),
    ]
    for column, band in enumerate(BAND_LABELS):
        x = left + column * cell_w + cell_w / 2
        parts.append(text(x, 86, band, 10, anchor="middle", weight="bold"))
        band_rows = rows[rows.band.eq(band)]
        if len(band_rows):
            item = band_rows.iloc[0]
            parts.append(text(x, 101, f"n={int(item.real_n)}/{int(item.synthetic_n)}", 9, anchor="middle", fill="#6b7280"))
    for row_index, feature in enumerate(FIGURE_FEATURES):
        y = top + row_index * cell_h
        parts.append(text(left - 10, y + 23, PROXY_LABELS[feature], 10, anchor="end", fill="#4b5563"))
        for column, band in enumerate(BAND_LABELS):
            x = left + column * cell_w
            item = rows[(rows.band == band) & (rows.feature == feature)]
            ratio = float(item.iloc[0].median_ratio) if len(item) else float("nan")
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{colour(ratio)}" stroke="white"/>')
            parts.append(text(x + (cell_w - 2) / 2, y + 22, "NA" if not np.isfinite(ratio) else f"{ratio:.2f}", 10, anchor="middle", fill="#111827", weight="bold"))
    parts.extend([
        text(24, height - 30, "Blue: synthetic lower than real; orange: synthetic higher. Ratios are descriptive, not a gate.", 10, fill="#4b5563"),
        "</svg>",
    ])
    path.write_text("\n".join(parts))


def write_png(rows: pd.DataFrame, path: Path) -> None:
    """Render the same heatmap as a portable raster figure for the PDF."""

    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_path = next((candidate for candidate in font_candidates if Path(candidate).is_file()), None)
    font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    small = ImageFont.truetype(font_path, 14) if font_path else ImageFont.load_default()
    bold = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default()
    image = Image.new("RGB", (1480, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), "Conditional proxy ratios by latitude band", fill="#1f2937", font=bold)
    draw.text((28, 48), "Synthetic median / real median; cell labels are ratios and n=real/synthetic rows", fill="#4b5563", font=small)
    left, top, cell_w, cell_h = 300, 125, 220, 42

    def fill(ratio: float) -> tuple[int, int, int]:
        if not np.isfinite(ratio):
            return (229, 231, 235)
        value = float(np.clip(np.log2(max(ratio, 1e-12)), -2.0, 2.0))
        if value < 0:
            strength = int(235 - 90 * abs(value) / 2.0)
            return strength, strength + 12, 255
        strength = int(235 - 90 * value / 2.0)
        return 255, strength + 18, strength

    for column, band in enumerate(BAND_LABELS):
        x = left + column * cell_w
        band_rows = rows[rows.band.eq(band)]
        item = band_rows.iloc[0]
        draw.text((x + cell_w / 2, 88), band, fill="#1f2937", font=small, anchor="mm")
        draw.text((x + cell_w / 2, 106), f"n={int(item.real_n)}/{int(item.synthetic_n)}", fill="#6b7280", font=small, anchor="mm")
    for row_index, feature in enumerate(FIGURE_FEATURES):
        y = top + row_index * cell_h
        draw.text((left - 12, y + cell_h / 2), PROXY_LABELS[feature], fill="#4b5563", font=small, anchor="rm")
        for column, band in enumerate(BAND_LABELS):
            x = left + column * cell_w
            item = rows[(rows.band == band) & (rows.feature == feature)]
            ratio = float(item.iloc[0].median_ratio) if len(item) else float("nan")
            draw.rectangle((x, y, x + cell_w - 3, y + cell_h - 3), fill=fill(ratio), outline="white", width=2)
            label = "NA" if not np.isfinite(ratio) else f"{ratio:.2f}"
            draw.text((x + (cell_w - 3) / 2, y + (cell_h - 3) / 2), label, fill="#111827", font=font, anchor="mm")
    draw.text((28, 520), "Blue: synthetic lower; orange: synthetic higher. Descriptive only; no gate or forecast claim.", fill="#4b5563", font=small)
    image.save(path, format="PNG", optimize=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    real = pd.read_csv(TABLE_DIR / "physics_proxy_real.csv")
    synthetic = pd.read_csv(TABLE_DIR / "physics_proxy_synthetic.csv")
    real["band"] = pd.cut(real.latitude_deg, BAND_EDGES, labels=BAND_LABELS, include_lowest=True).astype(str)
    synthetic["band"] = pd.cut(synthetic.latitude_deg, BAND_EDGES, labels=BAND_LABELS, include_lowest=True).astype(str)
    rows: list[dict[str, object]] = []
    for band_index, band in enumerate(BAND_LABELS):
        real_band = real[real.band.eq(band)]
        synthetic_band = synthetic[synthetic.band.eq(band)]
        for feature_index, feature in enumerate(PROXY_COLUMNS):
            ratio, ci_low, ci_high = bootstrap_ratio(
                real_band,
                synthetic_band,
                feature,
                args.seed + band_index * 100 + feature_index,
                args.bootstrap_reps,
            )
            rows.append(
                {
                    "band": band,
                    "feature": feature,
                    "real_n": int(len(real_band)),
                    "synthetic_n": int(len(synthetic_band)),
                    "real_groups": int(real_band.region_group_id.nunique()),
                    "synthetic_groups": int(synthetic_band.region_group_id.nunique()),
                    "real_median": float(real_band[feature].median()) if len(real_band) else float("nan"),
                    "synthetic_median": float(synthetic_band[feature].median()) if len(synthetic_band) else float("nan"),
                    "median_ratio": ratio,
                    "ratio_ci_low": ci_low,
                    "ratio_ci_high": ci_high,
                }
            )
    result = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(TABLE_DIR / "conditional_proxy_bands.csv", index=False)
    summary = result[result.feature.isin(FIGURE_FEATURES)].copy()
    summary.to_csv(TABLE_DIR / "conditional_proxy_summary.csv", index=False)
    with (TABLE_DIR / "conditional_proxy_summary.tex").open("w") as handle:
        handle.write("\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n")
        handle.write("\\begin{longtable}{p{0.16\\textwidth}p{0.20\\textwidth}rrrrrr}\n")
        handle.write("\\caption{Conditional proxy ratios by latitude band.}\\label{tab:conditional-proxies}\\\\\n")
        handle.write("\\toprule\nBand & Feature & real $n$ & synthetic $n$ & real groups & synthetic groups & ratio & 95\\% interval\\\\\n")
        handle.write("\\midrule\n\\endfirsthead\\toprule\nBand & Feature & real $n$ & synthetic $n$ & real groups & synthetic groups & ratio & 95\\% interval\\\\\n\\midrule\\endhead\n")
        for row in summary.itertuples(index=False):
            interval = "NA" if not np.isfinite(row.ratio_ci_low) else f"{row.ratio_ci_low:.2f}--{row.ratio_ci_high:.2f}"
            band_label = r"\texttt{" + str(row.band) + "}"
            handle.write(
                f"{band_label} & {PROXY_LABELS[row.feature]} & {row.real_n} & {row.synthetic_n} & "
                f"{row.real_groups} & {row.synthetic_groups} & {row.median_ratio:.3f} & {interval}\\\\\n"
            )
        handle.write("\\bottomrule\n\\end{longtable}\n\\normalsize\n\\setlength{\\tabcolsep}{6pt}\n")
    write_svg(summary, FIGURE_DIR / "conditional_proxy_heatmap.svg")
    write_png(summary, FIGURE_DIR / "conditional_proxy_heatmap.png")
    payload = {
        "status": "PASS",
        "method": "fixed latitude bands with independent connected-region bootstrap of medians",
        "seed": int(args.seed),
        "bootstrap_replicates": int(args.bootstrap_reps),
        "real_count": int(len(real)),
        "synthetic_count": int(len(synthetic)),
        "real_groups": int(real.region_group_id.nunique()),
        "synthetic_groups": int(synthetic.region_group_id.nunique()),
        "bands": [
            {
                "band": band,
                "real_n": int((real.band == band).sum()),
                "synthetic_n": int((synthetic.band == band).sum()),
                "real_groups": int(real.loc[real.band == band, "region_group_id"].nunique()),
                "synthetic_groups": int(synthetic.loc[synthetic.band == band, "region_group_id"].nunique()),
            }
            for band in BAND_LABELS
        ],
        "outputs": [
            "tables/conditional_proxy_bands.csv",
            "tables/conditional_proxy_summary.csv",
            "tables/conditional_proxy_summary.tex",
            "figures/conditional_proxy_heatmap.svg",
            "figures/conditional_proxy_heatmap.png",
        ],
        "limitations": [
            "The upper-latitude band has very few rows and should not be overinterpreted.",
            "These are descriptive train-only diagnostics and do not modify the frozen generic gate.",
            "Conditional proxy agreement would not establish physical realism or forecasting utility.",
        ],
    }
    (ARTIFACT_DIR / "conditional_proxy_diagnostics.json").write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
