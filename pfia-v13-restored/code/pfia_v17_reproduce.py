"""PFIA v17 deepened reproduction script.

Run from package root:
    python code/pfia_v17_reproduce.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def progression_null(top, candidates_per_year_values=(10, 25, 50, 100), sims=5000, seed=20260503):
    rng = np.random.default_rng(seed)
    X = sm.add_constant(top[["year"]])
    mod = sm.OLS(top["seconds"], X).fit()
    alpha = float(mod.params["const"])
    beta = float(mod.params["year"])
    sigma = float(mod.resid.std(ddof=1))
    years = np.arange(1967, 2027)
    rows = []
    for cpy in candidates_per_year_values:
        counts = []
        for _ in range(sims):
            yy = np.repeat(years, cpy)
            sec = alpha + beta * yy + rng.normal(0, sigma, len(yy))
            idx = np.argsort(sec)[:100]
            counts.append(int((yy[idx] >= 2018).sum()))
        arr = np.array(counts)
        rows.append({
            "candidates_per_year": cpy,
            "simulations": sims,
            "mean_count_2018_onward": arr.mean(),
            "sd_count_2018_onward": arr.std(ddof=1),
            "p_ge_observed_79": float((arr >= 79).mean()),
            "q025": float(np.quantile(arr, .025)),
            "median": float(np.quantile(arr, .5)),
            "q975": float(np.quantile(arr, .975)),
            "trend_slope_seconds_per_year": beta,
            "residual_sigma_seconds": sigma,
        })
    return pd.DataFrame(rows)


def main():
    records = pd.read_csv(DATA / "marathon_world_record_progression.csv")
    records = records[records["year_decimal"] >= 1967].copy()
    records["after_2017"] = np.maximum(0, records["year_decimal"] - 2017)

    linear = sm.OLS(records["seconds"], sm.add_constant(records[["year_decimal"]])).fit()
    segmented = sm.OLS(records["seconds"], sm.add_constant(records[["year_decimal", "after_2017"]])).fit()

    print("Marathon linear slope", round(linear.params["year_decimal"], 4))
    print("Marathon linear AIC", round(linear.aic, 4))
    print("Marathon segmented pre-2017 slope", round(segmented.params["year_decimal"], 4))
    print("Marathon segmented slope change", round(segmented.params["after_2017"], 4))
    print("Marathon segmented post-2017 slope", round(segmented.params["year_decimal"] + segmented.params["after_2017"], 4))
    print("Marathon segmented p", segmented.pvalues["after_2017"])

    print("\nTop-list compression")
    print(pd.read_csv(DATA / "toplist_compression_v13.csv").to_string(index=False))

    print("\nSmooth-progression top-list null sensitivity")
    top = pd.read_csv(DATA / "marathon_top100_partial_2026_update_clean.csv")
    null = progression_null(top)
    print(null.to_string(index=False))

    print("\nATP endpoint summary")
    print(pd.read_csv(DATA / "atp_change_summary.csv").to_string(index=False))

    print("\nATP surface-interaction model")
    print(pd.read_csv(DATA / "atp_surface_interaction_model_v13.csv").to_string(index=False))


if __name__ == "__main__":
    main()
