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
from iris_report.iris_sep.tools.render_availability_fallback_judge_summary import (
    render_markdown,
    validate_summary,
)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def judge_summary_fixture():
    scenarios = {}
    mapping = {
        "NO_XRS": "XRS",
        "NO_PROTON": "PROTON",
        "NO_XRS_OR_PROTON": "XRS_AND_PROTON",
    }
    passes = {
        "NO_XRS": (True, True, True),
        "NO_PROTON": (True, False, True),
        "NO_XRS_OR_PROTON": (False, False, False),
    }
    for state, modality in mapping.items():
        for hours, passed in zip((24, 72, 168), passes[state]):
            scenarios[f"{modality}_{hours}H"] = {
                "availability_state": state,
                "operator_gate": {"passed": passed},
            }
    return {
        "locked_test_accessed": False,
        "monitor_used": False,
        "imputation_used": False,
        "reconstruction_used": False,
        "retraining_at_outage_time": False,
        "states": {
            "FULL": {
                "whole_score": {
                    "MAX_TSS": {
                        "rows": 100,
                        "positives": 5,
                        "TSS": 0.3,
                        "POD": 0.6,
                        "FAR": 0.9,
                        "BRIER": 0.1,
                        "ECE": 0.05,
                    }
                }
            }
        },
        "scenarios": scenarios,
        "operator_state_decision": {
            "NO_XRS": {
                "permission": "DEGRADED",
                "normal_allowed": False,
                "all_three_durations_pass_degraded_candidate_gate": True,
            },
            "NO_PROTON": {
                "permission": "ABSTAIN",
                "normal_allowed": False,
                "all_three_durations_pass_degraded_candidate_gate": False,
            },
            "NO_XRS_OR_PROTON": {
                "permission": "ABSTAIN",
                "normal_allowed": False,
                "all_three_durations_pass_degraded_candidate_gate": False,
            },
        },
    }


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

    def test_judge_summary_surfaces_failures_and_false_alarm_rate(self):
        text = render_markdown(judge_summary_fixture())
        self.assertIn("X-ray feed unavailable | PASS | PASS | PASS | DEGRADED | no", text)
        self.assertIn("proton-context feed unavailable | PASS | FAIL | PASS | ABSTAIN | no", text)
        self.assertIn("FAR 0.900", text)
        self.assertIn("Locked test accessed: **no**", text)
        self.assertIn("cannot establish operational superiority", text)

    def test_judge_summary_refuses_to_hide_locked_test_access(self):
        summary = judge_summary_fixture()
        summary["locked_test_accessed"] = True
        with self.assertRaises(ValueError):
            validate_summary(summary)


if __name__ == "__main__":
    unittest.main()
