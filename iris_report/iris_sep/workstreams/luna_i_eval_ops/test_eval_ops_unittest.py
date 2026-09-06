"""Synthetic-only tests for evaluation and operator contracts."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

import numpy as np

from .evaluation import (
    EvaluationError,
    apply_calibration,
    fit_intercept_calibration,
    minimum_far_at_pod,
    paired_unit_bootstrap_tss_difference,
    probability_metrics,
    select_tss_threshold,
    threshold_metrics,
)
from .operator import OperatorRuntimePolicy, build_operator_forecast


class EvaluationTests(unittest.TestCase):
    def test_calibration_and_threshold_roles_fail_closed(self) -> None:
        with self.assertRaises(EvaluationError):
            fit_intercept_calibration([0, 1], [0, 1], role="train")
        calibration = fit_intercept_calibration([-1, -0.2, 0.2, 1], [0, 0, 1, 1], role="validation_calibration")
        probability = apply_calibration([-1, 1], calibration)
        self.assertTrue(np.all((probability > 0) & (probability < 1)))
        with self.assertRaises(EvaluationError):
            select_tss_threshold([0, 1], [0.1, 0.9], role="locked_test")

    def test_perfect_classifier_metrics(self) -> None:
        y = [0, 0, 1, 1]; p = [0.1, 0.2, 0.8, 0.9]
        threshold = select_tss_threshold(y, p, role="validation_threshold")
        scores = threshold_metrics(y, p, threshold.threshold)
        self.assertAlmostEqual(scores["TSS"], 1.0)
        probability = probability_metrics(y, p, reference_probability=0.5)
        self.assertAlmostEqual(probability["AUROC"], 1.0)
        self.assertAlmostEqual(probability["AUPRC"], 1.0)

    def test_probability_metrics_are_invariant_to_order_with_tied_scores(self) -> None:
        first = probability_metrics([1, 0, 1, 0], [0.8, 0.8, 0.2, 0.2], reference_probability=0.5)
        second = probability_metrics([0, 1, 0, 1], [0.8, 0.8, 0.2, 0.2], reference_probability=0.5)
        self.assertAlmostEqual(first["AUPRC"], second["AUPRC"])
        self.assertAlmostEqual(first["AUROC"], second["AUROC"])

    def test_matched_pod_is_deterministic(self) -> None:
        result = minimum_far_at_pod([1, 1, 0, 0], [0.9, 0.7, 0.8, 0.1], 1.0)
        self.assertEqual(result["achieved_POD"], 1.0)
        self.assertEqual(result["FAR"], 1 / 3)

    def test_paired_bootstrap_uses_units_and_detects_superior_scores(self) -> None:
        labels = [1, 0] * 10
        iris = [value for _ in range(10) for value in (0.9, 0.1)]
        baseline = [value for _ in range(10) for value in (0.4, 0.6)]
        units = [f"unit-{index}" for index in range(20)]
        result = paired_unit_bootstrap_tss_difference(
            labels, iris, baseline, units,
            iris_threshold=0.5, comparator_threshold=0.5,
            replicates=500, seed=7,
        )
        self.assertGreater(result["median_difference"], 0)

    def test_operator_output_abstains_on_critical_missing_input(self) -> None:
        forecast = build_operator_forecast(
            issued_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            calibrated_probability=0.8,
            runtime_policy=OperatorRuntimePolicy(
                policy_id="policy-v1", calibration_id="cal-v1", schema_sha256="b" * 64,
                operating_thresholds={"MONITOR": 0.2, "PREPARE": 0.5, "PROTECT": 0.75},
                maximum_age_minutes={"magnetic": 120, "eruption": 60, "particle_context": 15},
                critical_modalities=("particle_context",),
            ),
            input_schema_sha256="b" * 64,
            data_freshness={},
            missing_modalities=["particle_context"],
            uncertainty={"between_seed_std": 0.03},
            model_version="model-v1",
            evidence_receipt_sha256="a" * 64,
        )
        self.assertEqual(forecast["forecast_status"], "ABSTAIN")
        self.assertIsNone(forecast["operator_state"])
        self.assertFalse(forecast["spacecraft_control"])

    def test_operator_abstains_on_schema_mismatch_or_stale_input(self) -> None:
        policy = OperatorRuntimePolicy(
            policy_id="policy-v1", calibration_id="cal-v1", schema_sha256="b" * 64,
            operating_thresholds={"MONITOR": 0.2, "PREPARE": 0.5, "PROTECT": 0.75},
            maximum_age_minutes={"magnetic": 120, "eruption": 60, "particle_context": 15},
            critical_modalities=("particle_context",),
        )
        common = dict(
            issued_at=datetime(2026, 1, 1, tzinfo=timezone.utc), calibrated_probability=0.8,
            runtime_policy=policy, missing_modalities=[], uncertainty={}, model_version="model-v1",
            evidence_receipt_sha256="a" * 64,
        )
        stale = build_operator_forecast(
            **common, input_schema_sha256="b" * 64,
            data_freshness={"magnetic": {"age_minutes": 10}, "eruption": {"age_minutes": 10}, "particle_context": {"age_minutes": 16}},
        )
        self.assertEqual(stale["forecast_status"], "ABSTAIN")
        self.assertIn("INPUT_TOO_STALE", stale["abstention_reasons"])
        mismatched = build_operator_forecast(
            **common, input_schema_sha256="c" * 64,
            data_freshness={"magnetic": {"age_minutes": 10}, "eruption": {"age_minutes": 10}, "particle_context": {"age_minutes": 10}},
        )
        self.assertIn("SCHEMA_FAILURE", mismatched["abstention_reasons"])


if __name__ == "__main__":
    unittest.main()
