from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from iris_report.iris_sep.src.iris_sep.sealed_evaluation import (
    SealedEvaluationError,
    build_forecast_seal,
    derive_new_crossing_labels,
    evaluate_sealed_cohort,
)


BASE = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
STATES = ("FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON")


def seal(issue, *, full=0.8, no_xrs=0.7, no_proton=0.6, solar=0.4, thresholds=None):
    thresholds = thresholds or {
        state: {"MAX_TSS": 0.5, "POD80_MIN_FAR": 0.4}
        for state in STATES
    }
    return build_forecast_seal(
        issued_at=issue,
        sealed_at=issue + timedelta(seconds=20),
        package_manifest_sha256=SHA_A,
        feature_row_sha256=SHA_B,
        source_authentication_sha256=SHA_C,
        causal_feature_derivation_sha256=SHA_D,
        probabilities={
            "FULL": full,
            "NO_XRS": no_xrs,
            "NO_PROTON": no_proton,
            "NO_XRS_OR_PROTON": solar,
        },
        thresholds=thresholds,
        operator_permissions={
            "FULL": "NORMAL_ONLY_IF_ADMISSION_PASSES",
            "NO_XRS": "DEGRADED",
            "NO_PROTON": "DEGRADED",
            "NO_XRS_OR_PROTON": "ABSTAIN",
        },
        architecture_id="fixture-v1",
    )


class SealedEvaluationTests(unittest.TestCase):
    def test_forecast_must_be_sealed_at_issue_time(self):
        with self.assertRaisesRegex(SealedEvaluationError, "five minutes"):
            build_forecast_seal(
                issued_at=BASE,
                sealed_at=BASE + timedelta(minutes=6),
                package_manifest_sha256=SHA_A,
                feature_row_sha256=SHA_B,
                source_authentication_sha256=SHA_C,
                causal_feature_derivation_sha256=SHA_D,
                probabilities={state: 0.1 for state in STATES},
                thresholds={state: {"MAX_TSS": 0.2, "POD80_MIN_FAR": 0.1} for state in STATES},
                operator_permissions={state: "DEGRADED" for state in STATES},
                architecture_id="fixture-v1",
            )

    def test_new_crossing_is_labeled_only_inside_24h_horizon(self):
        first = seal(BASE)
        second = seal(BASE + timedelta(days=2))
        times = [
            BASE - timedelta(minutes=5),
            BASE + timedelta(hours=12),
            BASE + timedelta(hours=25),
            BASE + timedelta(days=2) - timedelta(minutes=5),
            BASE + timedelta(days=2, hours=25),
        ]
        flux = [2.0, 12.0, 15.0, 2.0, 12.0]
        labels = derive_new_crossing_labels(
            forecast_seals=[first, second],
            proton_times=times,
            proton_flux=flux,
        )
        self.assertEqual(labels["rows"][0]["label"], 1)
        self.assertEqual(labels["rows"][1]["label"], 0)

    def test_issue_already_above_threshold_is_ineligible(self):
        s = seal(BASE)
        labels = derive_new_crossing_labels(
            forecast_seals=[s],
            proton_times=[BASE - timedelta(minutes=5), BASE + timedelta(hours=1)],
            proton_flux=[11.0, 12.0],
        )
        row = labels["rows"][0]
        self.assertFalse(row["eligible_new_crossing_issue"])
        self.assertIsNone(row["label"])

    def test_support_gate_prevents_tiny_cohort_from_being_called_final(self):
        forecasts = [seal(BASE + timedelta(days=i), full=0.9 if i == 0 else 0.1) for i in range(3)]
        times = []
        flux = []
        for i in range(3):
            issue = BASE + timedelta(days=i)
            times.extend([issue - timedelta(minutes=5), issue + timedelta(hours=1)])
            flux.extend([1.0, 12.0 if i == 0 else 1.0])
        labels = derive_new_crossing_labels(forecast_seals=forecasts, proton_times=times, proton_flux=flux)
        result = evaluate_sealed_cohort(
            forecast_seals=forecasts,
            label_receipt=labels,
            minimum_positive_support=2,
        )
        self.assertEqual(result["positives"], 1)
        self.assertFalse(result["support_gate_passed"])
        self.assertIn("INSUFFICIENT_POSITIVE_SUPPORT", result["claim_status"])

    def test_frozen_threshold_cannot_change_between_forecasts(self):
        first = seal(BASE)
        changed = {state: {"MAX_TSS": 0.5, "POD80_MIN_FAR": 0.4} for state in STATES}
        changed["FULL"] = {"MAX_TSS": 0.6, "POD80_MIN_FAR": 0.4}
        second = seal(BASE + timedelta(days=1), thresholds=changed)
        labels = derive_new_crossing_labels(
            forecast_seals=[first, second],
            proton_times=[
                BASE - timedelta(minutes=5), BASE + timedelta(hours=1),
                BASE + timedelta(days=1) - timedelta(minutes=5), BASE + timedelta(days=1, hours=1),
            ],
            proton_flux=[1.0, 12.0, 1.0, 12.0],
        )
        with self.assertRaisesRegex(SealedEvaluationError, "threshold changed"):
            evaluate_sealed_cohort(
                forecast_seals=[first, second],
                label_receipt=labels,
                minimum_positive_support=1,
            )

    def test_derivation_digest_is_part_of_seal_integrity(self):
        original = seal(BASE)
        mutated = dict(original)
        mutated["causal_feature_derivation_sha256"] = "e" * 64
        with self.assertRaisesRegex(SealedEvaluationError, "seal digest mismatch"):
            derive_new_crossing_labels(
                forecast_seals=[mutated],
                proton_times=[BASE - timedelta(minutes=5), BASE + timedelta(hours=1)],
                proton_flux=[1.0, 2.0],
            )


if __name__ == "__main__":
    unittest.main()
