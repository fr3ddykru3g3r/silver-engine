"""
PFIA v12 reproducibility script.

Run from repository root:
    python pfia-v12/code/pfia_v12_reproduce.py

The script recomputes the marathon frontier models, top-list compression intervals,
ATP endpoint summaries, and the ATP surface-interaction model used as the PFIA
prediction check.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest
from statsmodels.stats.stattools import durbin_watson

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
BASE = HERE.parents[1] / "data"
FALLBACK_BASE = ROOT / "pfma-reproducibility" / "data"


def data_path(filename: str) -> Path:
    primary = BASE / filename
    if primary.exists():
        return primary
    return FALLBACK_BASE / filename


def fit_segmented(wr: pd.DataFrame, breakpoint_year: int = 2017):
    wr = wr.copy()
    wr["after_2017"] = np.maximum(0, wr["year"] - breakpoint_year)
    return sm.OLS(wr["seconds"], sm.add_constant(wr[["year", "after_2017"]])).fit()


def wilson(k: int, n: int, z: float = 1.959963984540054):
    ph = k / n
    denom = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / denom
    half = z * ((ph * (1 - ph) + z * z / (4 * n)) / n) ** 0.5 / denom
    return centre - half, centre + half


def main() -> None:
    wr = pd.read_csv(data_path("marathon_world_records_modern.csv"))
    if "year" not in wr.columns:
        wr["year"] = wr["year_decimal"]

    linear = sm.OLS(wr["seconds"], sm.add_constant(wr["year"])).fit()
    segmented = fit_segmented(wr, 2017)

    print("Marathon frontier models")
    print("------------------------")
    print(f"Linear slope: {linear.params['year']:.6f} s/year")
    print(f"Linear AIC: {linear.aic:.6f}")
    print(f"Segmented pre-2017 slope: {segmented.params['year']:.6f} s/year")
    print(f"Segmented slope change: {segmented.params['after_2017']:.6f} s/year")
    print(f"Segmented post-2017 slope: {(segmented.params['year'] + segmented.params['after_2017']):.6f} s/year")
    print(f"Segmented p-value for slope change: {segmented.pvalues['after_2017']:.12f}")
    print(f"Segmented AIC: {segmented.aic:.6f}")
    print(f"Segmented residual Durbin-Watson: {durbin_watson(segmented.resid):.6f}")

    print("\nTop-list compression intervals")
    print("------------------------------")
    try:
        ci = pd.read_csv(data_path("toplist_compression_ci_v12.csv"))
        print(ci.to_string(index=False))
    except FileNotFoundError:
        top = pd.read_csv(data_path("marathon_top100_partial_2026.csv")).sort_values("seconds")
        rows = []
        for cutoff in [10, 25, 50, 100]:
            sub = top.iloc[:cutoff]
            k = int(sub["post_2017"].sum())
            lo, hi = wilson(k, len(sub))
            rows.append({
                "rank_cutoff": cutoff,
                "post_2017_count": k,
                "n": len(sub),
                "share": k / len(sub),
                "wilson_low": lo,
                "wilson_high": hi,
                "binom_p_calendar": binomtest(k, len(sub), 9 / 60, alternative="greater").pvalue,
            })
        print(pd.DataFrame(rows).to_string(index=False))

    print("\nATP endpoint summary")
    print("--------------------")
    print(pd.read_csv(data_path("atp_change_summary.csv")).to_string(index=False))

    print("\nATP surface-interaction model")
    print("-----------------------------")
    surface = pd.read_csv(data_path("atp_surface_annual_1991_2024.csv"))
    surface = surface[surface["surface"].isin(["Clay", "Grass", "Hard"])].copy()
    surface["year_c"] = surface["year"] - 1991
    model = smf.ols("ace_rate_per_100_svpt ~ year_c * C(surface)", data=surface).fit()
    print(model.params.to_string())
    print("\np-values")
    print(model.pvalues.to_string())


if __name__ == "__main__":
    main()
