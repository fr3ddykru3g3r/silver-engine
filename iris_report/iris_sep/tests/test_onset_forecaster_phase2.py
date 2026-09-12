from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools.run_onset_forecaster_phase2 import (
    CANDIDATE_ORDER,
    REGULAR_BASELINES,
    build_feature_frames,
    event_balanced_weights,
)


class Phase2WeightingTests(unittest.TestCase):
    def test_event_balanced_weights_equalize_classes_and_units(self):
        # Positive event p1 has two rows, p2 one row. Negative quiet block n1
        # has three rows, n2 one row. Unit duration must not change total mass.
        y = np.asarray([1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
        units = np.asarray(["p1", "p1", "p2", "n1", "n1", "n1", "n2"])
        idx = np.arange(len(y), dtype=int)
        w = event_balanced_weights(y, units, idx)
        self.assertAlmostEqual(float(np.mean(w)), 1.0, places=12)
        self.assertAlmostEqual(float(w[y == 1].sum()), float(w[y == 0].sum()), places=12)
        for label in (0, 1):
            totals = []
            for unit in np.unique(units[y == label]):
                totals.append(float(w[(units == unit) & (y == label)].sum()))
            self.assertAlmostEqual(min(totals), max(totals), places=12)

    def test_event_balanced_weights_reject_mixed_label_unit(self):
        y = np.asarray([1, 0, 0], dtype=np.int8)
        units = np.asarray(["bad", "bad", "quiet"])
        with self.assertRaisesRegex(ValueError, "mixes labels"):
            event_balanced_weights(y, units, np.arange(3, dtype=int))

    def test_event_balanced_weights_require_both_classes(self):
        y = np.asarray([0, 0], dtype=np.int8)
        units = np.asarray(["a", "b"])
        with self.assertRaisesRegex(ValueError, "both classes"):
            event_balanced_weights(y, units, np.arange(2, dtype=int))


class Phase2FeatureContractTests(unittest.TestCase):
    def test_forbidden_catalogue_label_cannot_enter_learned_features(self):
        frame = pd.DataFrame({
            "window_end": pd.to_datetime(["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"]),
            "solar": [1.0, 2.0],
            "xrs": [0.1, 0.2],
            "protonflux": [0.3, 0.4],
            "OSEP_label": [0, 1],
        })
        with self.assertRaisesRegex(ValueError, "forbidden learned feature"):
            build_feature_frames(frame, ["solar", "OSEP_label"], ["xrs"], ["protonflux"])

    def test_preregistration_matches_frozen_candidate_sets(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config" / "onset_forecaster_phase2_preregistration_2026-09-11.json").read_text())
        self.assertEqual(config["fixed_candidate_order"], list(CANDIDATE_ORDER))
        self.assertEqual(config["regular_baselines"], list(REGULAR_BASELINES))
        self.assertEqual(config["inner_selection"]["candidate_count"], len(CANDIDATE_ORDER))
        self.assertFalse(config["protected_outcomes"]["locked_test_accessed"])
        self.assertEqual(config["protected_outcomes"]["post_2025_pool_access"], "FORBIDDEN")


if __name__ == "__main__":
    unittest.main()
