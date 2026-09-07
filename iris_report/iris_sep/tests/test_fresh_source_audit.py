from __future__ import annotations

import unittest

from iris_report.iris_sep.tools.run_fresh_source_audit import summarize_protons, summarize_xrs


class FreshSourceAuditTests(unittest.TestCase):
    def test_proton_parser_selects_operational_channel_and_threshold(self):
        records = [
            {"time_tag": "2026-09-01T00:00:00Z", "energy": ">=10 MeV", "flux": 1.0},
            {"time_tag": "2026-09-01T00:05:00Z", "energy": ">=10 MeV", "flux": 12.0},
            {"time_tag": "2026-09-01T00:00:00Z", "energy": ">=50 MeV", "flux": 0.1},
        ]
        result = summarize_protons(records)
        self.assertEqual(result["channel"], ">=10 MeV")
        self.assertEqual(result["samples_at_or_above_10_pfu"], 1)
        self.assertTrue(result["threshold_10_pfu_active_at_latest_sample"])
        self.assertEqual(result["cadence"]["median_cadence_seconds"], 300.0)

    def test_xrs_parser_keeps_bands_separate(self):
        records = [
            {"time_tag": "2026-09-01T00:00:00Z", "energy": "0.05-0.4nm", "flux": 1e-7},
            {"time_tag": "2026-09-01T00:01:00Z", "energy": "0.05-0.4nm", "flux": 2e-7},
            {"time_tag": "2026-09-01T00:00:00Z", "energy": "0.1-0.8nm", "flux": 2e-6},
            {"time_tag": "2026-09-01T00:01:00Z", "energy": "0.1-0.8nm", "flux": 3e-6},
        ]
        result = summarize_xrs(records)
        self.assertEqual(set(result["bands"]), {"0.05-0.4nm", "0.1-0.8nm"})
        self.assertEqual(result["bands"]["0.1-0.8nm"]["cadence"]["median_cadence_seconds"], 60.0)
        self.assertAlmostEqual(result["bands"]["0.1-0.8nm"]["max_flux_w_m2"], 3e-6)

    def test_empty_sources_fail_closed(self):
        with self.assertRaises(ValueError):
            summarize_protons([])
        with self.assertRaises(ValueError):
            summarize_xrs([])


if __name__ == "__main__":
    unittest.main()
