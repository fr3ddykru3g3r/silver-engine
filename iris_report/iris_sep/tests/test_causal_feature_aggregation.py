from __future__ import annotations

from datetime import datetime, timezone
import math
import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.src.iris_sep.causal_feature_aggregation import (
    CausalFeatureAggregationError,
    build_causal_feature_row,
)


ISSUE = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def schema():
    solar = [
        "SHARP_label", "SHARP_From_SMARP_label",
        "SHARP_USFLUX_min", "SHARP_USFLUX_max", "SHARP_USFLUX_avg",
        "SHARP_AR_label", "SHARP_AR_From_SMARP_label",
        "SHARP_AR_USFLUX_min", "SHARP_AR_USFLUX_max", "SHARP_AR_USFLUX_avg",
        "Flare_label", "Flare_num",
        "Flare_Duration_min", "Flare_Duration_max", "Flare_Duration_avg",
        "Flare_RisenTime_min", "Flare_RisenTime_max", "Flare_RisenTime_avg",
        "Flare_log_Strength_min", "Flare_log_Strength_max", "Flare_log_Strength_avg",
        "DONKICME_label", "DONKICME_num", "DONKICME_From_CDAW_label",
        "DONKICME_lon_deg_min", "DONKICME_lon_deg_max", "DONKICME_lon_deg_avg",
        "DONKICME_halfAngle_deg_min", "DONKICME_halfAngle_deg_max", "DONKICME_halfAngle_deg_avg",
        "DONKICME_speed_km_s_min", "DONKICME_speed_km_s_max", "DONKICME_speed_km_s_avg",
        "CDAWCME_label", "CDAWCME_num",
        "CDAWCME_central_pa_deg_min", "CDAWCME_central_pa_deg_max", "CDAWCME_central_pa_deg_avg",
        "CDAWCME_width_deg_min", "CDAWCME_width_deg_max", "CDAWCME_width_deg_avg",
        "CDAWCME_speed_km_s_min", "CDAWCME_speed_km_s_max", "CDAWCME_speed_km_s_avg",
        "CDAWCME_energy_erg_min", "CDAWCME_energy_erg_max", "CDAWCME_energy_erg_avg",
    ]
    return {
        "solar": solar,
        "xrs": ["XRS_label", "XRSB_min", "XRSB_max", "XRSB_avg"],
        "proton": ["ProtonFlux_label", "ProtonFlux_min", "ProtonFlux_max", "ProtonFlux_avg"],
    }


class CausalFeatureAggregationTests(unittest.TestCase):
    def sources(self):
        sharp = pd.DataFrame({
            "T_REC": ["2026-09-06T13:00:00Z", "2026-09-07T11:00:00Z"],
            "NOAA_AR": [14500, 14501],
            "USFLUX": [10.0, 30.0],
        })
        flares = pd.DataFrame({
            "start_time": ["2026-09-07T10:00:00Z"],
            "peak_time": ["2026-09-07T10:10:00Z"],
            "end_time": ["2026-09-07T10:30:00Z"],
            "label": ["M2.0"],
            "ar_noaanum": [14501],
        })
        donki = pd.DataFrame({
            "startTime": ["2026-09-07T09:00:00Z"],
            "lon_deg": [15.0],
            "halfAngle_deg": [30.0],
            "speed_km_s": [900.0],
        })
        cdaw = pd.DataFrame({
            "cme_time": ["2026-09-07T08:00:00Z"],
            "central_pa_deg": [180.0],
            "width_deg": [120.0],
            "speed_km_s": [850.0],
            "energy_erg": [1e31],
        })
        proton = pd.DataFrame({
            "time": ["2026-09-07T10:00:00Z", "2026-09-07T11:00:00Z"],
            "flux": [1.0, 3.0],
        })
        xrs = pd.DataFrame({
            "time": ["2026-09-07T10:00:00Z", "2026-09-07T11:00:00Z"],
            "xrsb": [1e-6, 4e-6],
        })
        return sharp, flares, donki, cdaw, proton, xrs

    def test_exact_past_only_aggregation_matches_frozen_semantics(self):
        sharp, flares, donki, cdaw, proton, xrs = self.sources()
        row = build_causal_feature_row(
            issue_time=ISSUE,
            feature_families=schema(),
            sharp=sharp,
            flares=flares,
            donki_cme=donki,
            cdaw_cme=cdaw,
            proton=proton,
            xrs=xrs,
        ).iloc[0]
        self.assertEqual(row["SHARP_label"], 1)
        self.assertEqual(row["SHARP_From_SMARP_label"], 0)
        self.assertAlmostEqual(row["SHARP_USFLUX_avg"], 20.0)
        self.assertEqual(row["SHARP_AR_label"], 1)
        self.assertAlmostEqual(row["SHARP_AR_USFLUX_avg"], 30.0)
        self.assertEqual(row["Flare_num"], 1)
        self.assertAlmostEqual(row["Flare_Duration_avg"], 30.0)
        self.assertAlmostEqual(row["Flare_RisenTime_avg"], 10.0)
        self.assertAlmostEqual(row["Flare_log_Strength_avg"], math.log10(2e-5))
        self.assertEqual(row["DONKICME_From_CDAW_label"], 0)
        self.assertAlmostEqual(row["ProtonFlux_avg"], 2.0)
        self.assertAlmostEqual(row["XRSB_max"], 4e-6)
        self.assertEqual(list(row.index), schema()["solar"] + schema()["xrs"] + schema()["proton"])

    def test_future_source_row_is_rejected_instead_of_silently_filtered(self):
        sharp, flares, donki, cdaw, proton, xrs = self.sources()
        proton.loc[len(proton)] = ["2026-09-07T12:00:01Z", 100.0]
        with self.assertRaisesRegex(CausalFeatureAggregationError, "after issue time"):
            build_causal_feature_row(
                issue_time=ISSUE,
                feature_families=schema(),
                sharp=sharp,
                flares=flares,
                donki_cme=donki,
                cdaw_cme=cdaw,
                proton=proton,
                xrs=xrs,
            )

    def test_empty_sources_use_only_frozen_empty_window_constants(self):
        empty = pd.DataFrame()
        row = build_causal_feature_row(
            issue_time=ISSUE,
            feature_families=schema(),
            sharp=empty,
            flares=empty,
            donki_cme=empty,
            cdaw_cme=empty,
            proton=empty,
            xrs=empty,
        ).iloc[0]
        self.assertEqual(row["SHARP_label"], 0)
        self.assertTrue(np.isnan(row["SHARP_USFLUX_avg"]))
        self.assertEqual(row["Flare_num"], 0)
        self.assertEqual(row["Flare_Duration_avg"], 0.0)
        self.assertEqual(row["Flare_log_Strength_avg"], -10.0)
        self.assertEqual(row["DONKICME_speed_km_s_avg"], 0.0)
        self.assertEqual(row["CDAWCME_speed_km_s_avg"], 0.0)
        self.assertEqual(row["ProtonFlux_label"], 0)
        self.assertEqual(row["ProtonFlux_avg"], 0.0)
        self.assertEqual(row["XRS_label"], 0)
        self.assertEqual(row["XRSB_avg"], 0.0)


if __name__ == "__main__":
    unittest.main()
