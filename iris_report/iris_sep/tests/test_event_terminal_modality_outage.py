from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools import run_event_terminal_modality_outage as mod


class EventTerminalOutageTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.Series(pd.date_range("2025-01-01", periods=25, freq="D", tz="UTC"))
        self.labels = np.zeros(25, dtype=int)
        self.labels[12] = 1
        self.score = np.ones(25, dtype=bool)
        self.finite = np.ones(25, dtype=bool)

    def test_event_selection_requires_positive_history_and_exposure(self):
        selected = mod.select_event_rows(
            self.times, self.labels, self.score, self.finite, self.times, 7
        )
        self.assertEqual(np.flatnonzero(selected).tolist(), [12])
        no_exposure = self.finite.copy(); no_exposure[12] = False
        selected2 = mod.select_event_rows(
            self.times, self.labels, self.score, no_exposure, self.times, 7
        )
        self.assertFalse(selected2.any())

    def test_quiet_control_is_unique_label_only_and_at_least_eight_days_away(self):
        events = mod.select_event_rows(
            self.times, self.labels, self.score, self.finite, self.times, 3
        )
        quiet = mod.select_quiet_controls(
            self.times, self.labels, self.score, self.finite, self.times, 3, events
        )
        self.assertEqual(int(quiet.sum()), 1)
        qi = int(np.flatnonzero(quiet)[0])
        self.assertEqual(self.labels[qi], 0)
        self.assertGreaterEqual(abs(self.times.iloc[qi] - self.times.iloc[12]), pd.Timedelta(days=8))

    def test_terminal_holdout_keeps_declared_gap_for_causal_recovery(self):
        values = np.arange(50, dtype=float).reshape(25, 2)
        terminal_rows = np.zeros(25, dtype=bool); terminal_rows[12] = True
        terminal, block = mod._terminal_and_block_holdout(
            values, self.times, terminal_rows, np.asarray([1]), 3
        )
        self.assertEqual(np.argwhere(terminal).tolist(), [[12, 1]])
        self.assertEqual(np.argwhere(block).tolist(), [[10, 1], [11, 1], [12, 1]])

    def test_identity_hash_is_order_invariant_but_identity_sensitive(self):
        a = mod.identity_sha256([self.times.iloc[1], self.times.iloc[2]])
        b = mod.identity_sha256([self.times.iloc[2], self.times.iloc[1]])
        c = mod.identity_sha256([self.times.iloc[1], self.times.iloc[3]])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
