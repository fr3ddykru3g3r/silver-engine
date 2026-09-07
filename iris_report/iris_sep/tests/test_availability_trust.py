from __future__ import annotations

from datetime import datetime, timezone
import unittest

from iris_report.iris_sep.src.iris_sep.modeling.availability_trust import (
    AvailabilityValidationEvidence,
    NormalPromotionRule,
    availability_permission,
)
from iris_report.iris_sep.workstreams.luna_i_eval_ops.operator import (
    OperatorRuntimePolicy,
    build_operator_forecast,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def evidence(**overrides):
    values = dict(
        availability_state="NO_PROTON",
        target="new_sep_10mev_10pfu_within_24h",
        evaluation_scope="INDEPENDENT_LOCKED_EVALUATION",
        evaluation_cohort_sha256=SHA_A,
        model_package_sha256=SHA_B,
        locked_configuration_sha256=SHA_C,
        development_rows_used=False,
        threshold_refit_on_evaluation=False,
        calibration_refit_on_evaluation=False,
        probabilities_finite=True,
        affected_tss=0.25,
        affected_pod=0.60,
        whole_score_brier_delta_vs_full=0.002,
        whole_score_ece_delta_vs_full=0.005,
        paired_tss_ci_lower_vs_full=-0.02,
        review_fraction=0.05,
        review_enrichment=6.0,
    )
    values.update(overrides)
    return AvailabilityValidationEvidence(**values)


def runtime_policy():
    return OperatorRuntimePolicy(
        policy_id="policy-v2",
        calibration_id="cal-v2",
        schema_sha256=SHA_B,
        operating_thresholds={"MONITOR": 0.20, "PREPARE": 0.50, "PROTECT": 0.75},
        maximum_age_minutes={"magnetic": 120, "eruption": 60, "particle_context": 15},
        critical_modalities=("particle_context",),
    )


class AvailabilityTrustTests(unittest.TestCase):
    def test_independent_noninferior_one_feed_fallback_can_be_valid(self):
        gate = availability_permission(evidence())
        self.assertEqual(gate["permission"], "VALID")
        self.assertTrue(gate["normal_allowed"])
        self.assertTrue(all(gate["normal_checks"].values()))

    def test_development_evidence_can_never_promote_normal(self):
        gate = availability_permission(
            evidence(evaluation_scope="DEVELOPMENT_ONLY", development_rows_used=True)
        )
        self.assertNotEqual(gate["permission"], "VALID")
        self.assertFalse(gate["normal_allowed"])

    def test_noninferiority_failure_downgrades_without_relabelling(self):
        gate = availability_permission(
            evidence(paired_tss_ci_lower_vs_full=-0.20)
        )
        self.assertEqual(gate["permission"], "DEGRADED")
        self.assertFalse(gate["normal_allowed"])
        self.assertFalse(gate["normal_checks"]["paired_TSS_noninferior"])

    def test_two_missing_context_families_cannot_promote_normal(self):
        gate = availability_permission(
            evidence(availability_state="NO_XRS_OR_PROTON")
        )
        self.assertNotEqual(gate["permission"], "VALID")
        self.assertFalse(gate["normal_checks"]["normal_state_eligible"])

    def test_operator_accepts_validated_fallback_for_critical_missing_modality(self):
        permission = availability_permission(evidence())["permission"]
        forecast = build_operator_forecast(
            issued_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            calibrated_probability=0.62,
            runtime_policy=runtime_policy(),
            input_schema_sha256=SHA_B,
            data_freshness={
                "magnetic": {"age_minutes": 10},
                "eruption": {"age_minutes": 5},
            },
            missing_modalities=["particle_context"],
            uncertainty={"between_seed_std": 0.02},
            model_version="availability-specialist-v2",
            evidence_receipt_sha256=SHA_A,
            availability_permission=str(permission),
        )
        self.assertEqual(forecast["forecast_status"], "VALID")
        self.assertEqual(forecast["availability_permission"], "VALID")
        self.assertEqual(forecast["operator_state"], "PREPARE")
        self.assertEqual(forecast["abstention_reasons"], [])

    def test_operator_remains_fail_closed_without_validation_permission(self):
        forecast = build_operator_forecast(
            issued_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            calibrated_probability=0.62,
            runtime_policy=runtime_policy(),
            input_schema_sha256=SHA_B,
            data_freshness={
                "magnetic": {"age_minutes": 10},
                "eruption": {"age_minutes": 5},
            },
            missing_modalities=["particle_context"],
            uncertainty={},
            model_version="availability-specialist-v2",
            evidence_receipt_sha256=SHA_A,
        )
        self.assertEqual(forecast["forecast_status"], "ABSTAIN")
        self.assertIn("CRITICAL_INPUT_MISSING", forecast["abstention_reasons"])

    def test_rule_validation_rejects_fake_review_budget(self):
        with self.assertRaises(ValueError):
            NormalPromotionRule(primary_review_fraction=0.0)


if __name__ == "__main__":
    unittest.main()
