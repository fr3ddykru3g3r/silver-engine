"""
PFMA v9 reproducibility script.

Run from the repository root:
    python pfma-reproducibility/code/pfma_v9_reproduce.py

The script recomputes the marathon linear and segmented models and prints ATP endpoint changes from processed CSV files.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = Path(__file__).resolve().parents[1] / "data"


def fit_marathon_models() -> None:
    wr = pd.read_csv(BASE / "marathon_world_records_modern.csv")
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


def print_atp_summaries() -> None:
    print("\nATP endpoint summary")
    print("--------------------")
    print(pd.read_csv(BASE / "atp_change_summary.csv").to_string(index=False))
    print("\nATP annual aggregate regression outputs")
    print("---------------------------------------")
    print(pd.read_csv(BASE / "atp_model_outputs.csv").to_string(index=False))


if __name__ == "__main__":
    fit_marathon_models()
    print_atp_summaries()
