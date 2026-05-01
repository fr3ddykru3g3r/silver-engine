"""
PFIA v11 reproducibility script.

Run from the repository root:
    python pfia-v11/code/pfia_v11_reproduce.py

The script recomputes the marathon linear and segmented models, reports a Durbin-Watson residual diagnostic, and prints ATP endpoint/regression summaries from processed CSV files.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
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


def fit_marathon_models() -> None:
    wr = pd.read_csv(data_path("marathon_world_records_modern.csv"))
    if "year" not in wr.columns:
        wr["year"] = wr["year_decimal"]

    linear = sm.OLS(wr["seconds"], sm.add_constant(wr["year"])).fit()

    wr["after_2017"] = np.maximum(0, wr["year"] - 2017)
    segmented = sm.OLS(wr["seconds"], sm.add_constant(wr[["year", "after_2017"]])).fit()

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


def print_atp_summaries() -> None:
    print("\nATP endpoint summary")
    print("--------------------")
    print(pd.read_csv(data_path("atp_change_summary.csv")).to_string(index=False))
    print("\nATP annual aggregate regression outputs")
    print("---------------------------------------")
    print(pd.read_csv(data_path("atp_model_outputs.csv")).to_string(index=False))


if __name__ == "__main__":
    fit_marathon_models()
    print_atp_summaries()
