"""PFIA v13 restored reproduction script.

Run from package root:
    python code/pfia_v13_reproduce.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


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

    print("\nATP endpoint summary")
    print(pd.read_csv(DATA / "atp_change_summary.csv").to_string(index=False))

    print("\nATP surface-interaction model")
    print(pd.read_csv(DATA / "atp_surface_interaction_model_v13.csv").to_string(index=False))


if __name__ == "__main__":
    main()
