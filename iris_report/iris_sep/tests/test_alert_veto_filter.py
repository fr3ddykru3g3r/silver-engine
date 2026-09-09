from __future__ import annotations

import unittest
import numpy as np

from iris_report.iris_sep.src.iris_sep.modeling.alert_veto_filter import (
    apply_veto,
    select_veto_threshold,
)


class AlertVetoFilterTests(unittest.TestCase):
    def test_veto_never_creates_new_alert(self):
        baseline = np.array([True, False, True, False, True])
        score = np.array([0.9, 0.99, 0.2, 0.8, 0.7])
        out = apply_veto(baseline_alert=baseline, decision_score=score, threshold=0.5)
        self.assertTrue(np.all(~out | baseline))
        self.assertFalse(out[1])
        self.assertFalse(out[3])

    def test_threshold_selection_minimizes_fp_with_tp_constraint_inside_baseline(self):
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        baseline = np.array([True, True, True, True, True, True, True, False])
        score = np.array([0.9, 0.8, 0.3, 0.7, 0.6, 0.2, 0.1, 0.99])
        result = select_veto_threshold(
            labels=y,
            decision_score=score,
            baseline_alert=baseline,
            max_true_positive_loss=1,
        )
        out = apply_veto(
            baseline_alert=baseline,
            decision_score=score,
            threshold=float(result['threshold']),
        )
        self.assertTrue(np.all(~out | baseline))
        self.assertGreaterEqual(result['true_positives'], 2)
        self.assertLessEqual(result['false_positives'], int(np.sum((y == 0) & baseline)))

    def test_no_veto_baseline_is_always_legal_candidate(self):
        y = np.array([1, 0, 0])
        baseline = np.array([True, True, False])
        score = np.array([0.1, 0.9, 0.8])
        result = select_veto_threshold(
            labels=y,
            decision_score=score,
            baseline_alert=baseline,
            max_true_positive_loss=0,
        )
        out = apply_veto(baseline_alert=baseline, decision_score=score, threshold=float(result['threshold']))
        self.assertTrue(out[0])
        self.assertTrue(np.all(~out | baseline))


if __name__ == '__main__':
    unittest.main()
