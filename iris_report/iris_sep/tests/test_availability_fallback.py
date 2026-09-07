from __future__ import annotations

import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.modeling.availability_fallback import (
    AvailabilityPromotionRule,
    evidence_columns,
    fit_availability_stacks,
    degraded_candidate_gate,
    select_evidence,
    stacked_raw_probability,
)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


class AvailabilityFallbackTests(unittest.TestCase):
    def setUp(self):
        self.evidence = np.array([
            [-1.0, -0.8, -0.6],
            [-0.7, -0.3, -0.4],
            [-0.2,  0.1, -0.1],
            [ 0.2,  0.6,  0.4],
            [ 0.8,  1.0,  0.7],
            [ 1.2,  1.4,  1.0],
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)

    def test_state_columns_are_fixed(self):
        self.assertEqual(evidence_columns("FULL"), (0, 1, 2))
        self.assertEqual(evidence_columns("NO_XRS"), (0, 2))
        self.assertEqual(evidence_columns("NO_PROTON"), (0, 1))
        self.assertEqual(evidence_columns("NO_XRS_OR_PROTON"), (0,))

    def test_selection_drops_missing_expert_without_imputation(self):
        x = select_evidence(self.evidence, "NO_XRS")
        np.testing.assert_array_equal(x, self.evidence[:, [0, 2]])

    def test_stacks_have_nonnegative_weights(self):
        stacks = fit_availability_stacks(self.evidence, self.y)
        self.assertEqual(set(stacks), {"FULL", "NO_XRS", "NO_PROTON"})
        for stack in stacks.values():
            self.assertTrue(all(w >= 0 for w in stack.fit_.weights))

    def test_solar_only_fallback_is_exact_solar_probability(self):
        stacks = fit_availability_stacks(self.evidence, self.y)
        solar = np.linspace(0.1, 0.6, len(self.y))
        p = stacked_raw_probability(
            state="NO_XRS_OR_PROTON",
            full_evidence=self.evidence,
            raw_solar_probability=solar,
            stacks=stacks,
            sigmoid=sigmoid,
        )
        np.testing.assert_array_equal(p, solar)

    def test_missing_stack_fails_closed(self):
        solar = np.linspace(0.1, 0.6, len(self.y))
        with self.assertRaises(ValueError):
            stacked_raw_probability(
                state="NO_XRS",
                full_evidence=self.evidence,
                raw_solar_probability=solar,
                stacks={},
                sigmoid=sigmoid,
            )

    def test_degraded_gate_passes_only_all_checks(self):
        good = degraded_candidate_gate(
            affected_tss=0.2,
            affected_pod=0.6,
            whole_score_brier_delta=0.001,
            whole_score_ece_delta=0.001,
            probabilities_finite=True,
        )
        self.assertTrue(good["passed"])
        self.assertEqual(good["permission"], "DEGRADED")
        self.assertFalse(good["normal_allowed"])

        bad = degraded_candidate_gate(
            affected_tss=-0.01,
            affected_pod=0.8,
            whole_score_brier_delta=0.0,
            whole_score_ece_delta=0.0,
            probabilities_finite=True,
        )
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["permission"], "ABSTAIN")

    def test_rule_rejects_invalid_margin(self):
        with self.assertRaises(ValueError):
            AvailabilityPromotionRule(maximum_whole_score_brier_delta=-0.1)


if __name__ == "__main__":
    unittest.main()
