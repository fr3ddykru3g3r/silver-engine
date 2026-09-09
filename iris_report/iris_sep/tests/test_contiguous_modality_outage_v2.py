from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools.run_contiguous_modality_outage_v2 import (
    deterministic_source_blocks,
    scenario_holdout,
)


class ContiguousModalityOutageV2Tests(unittest.TestCase):
    def setUp(self):
        self.source = pd.Series(pd.date_range("2024-01-01T00:00:00Z", periods=3000, freq="h"))

    def test_blocks_are_hourly_duration_and_nonoverlapping(self):
        blocks = deterministic_source_blocks(
            self.source,
            168,
            self.source.iloc[100],
            self.source.iloc[2800],
        )
        self.assertEqual(len(blocks), 5)
        for block in blocks:
            self.assertEqual(
                block["issue_end_exclusive"] - block["issue_start"],
                pd.Timedelta(hours=168),
            )
        ordered = sorted(blocks, key=lambda b: b["issue_start"])
        for left, right in zip(ordered, ordered[1:]):
            self.assertLessEqual(left["issue_end_exclusive"], right["issue_start"])

    def test_source_clock_not_filtered_score_clock_controls_placement(self):
        # Eligible rows deliberately contain multi-hour gaps, reproducing the V1
        # failure mode. The underlying source clock remains perfectly hourly.
        eligible = self.source.iloc[200:2200:3].reset_index(drop=True)
        n = len(eligible)
        observed = np.ones((n, 4), dtype=bool)
        score = np.ones(n, dtype=bool)
        feature_indices = np.array([1, 3], dtype=int)

        holdout, blocks = scenario_holdout(
            observed,
            score,
            eligible,
            self.source,
            feature_indices,
            72,
        )

        self.assertEqual(len(blocks), 5)
        self.assertTrue(holdout.any())
        self.assertTrue(all(b["source_duration_hours"] == 72 for b in blocks))
        # Only the requested modality columns can be hidden.
        self.assertFalse(holdout[:, 0].any())
        self.assertFalse(holdout[:, 2].any())
        self.assertTrue(holdout[:, 1].any())
        self.assertTrue(holdout[:, 3].any())

    def test_projection_never_hides_non_score_rows(self):
        eligible = self.source.iloc[100:1800:2].reset_index(drop=True)
        n = len(eligible)
        observed = np.ones((n, 3), dtype=bool)
        score = np.zeros(n, dtype=bool)
        score[n // 4:] = True
        holdout, _ = scenario_holdout(
            observed,
            score,
            eligible,
            self.source,
            np.array([0, 1], dtype=int),
            24,
        )
        self.assertFalse(holdout[~score].any())

    def test_block_selection_has_no_label_argument(self):
        # A regression guard against accidentally making placement outcome-aware:
        # the same source/role timestamps necessarily return identical blocks.
        a = deterministic_source_blocks(
            self.source, 24, self.source.iloc[200], self.source.iloc[2500]
        )
        b = deterministic_source_blocks(
            self.source, 24, self.source.iloc[200], self.source.iloc[2500]
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
