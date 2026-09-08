from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import unittest

from iris_report.iris_sep.src.iris_sep.sealed_evaluation import (
    LABEL_FORMAT,
    SealedEvaluationError,
    build_forecast_seal,
    derive_new_crossing_labels,
    evaluate_sealed_cohort,
    validate_forecast_seal,
)


BASE = datetime(2026, 9, 7, 0, 0, tzinfo=timezone.utc)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
STATES = ("FULL", "NO_XRS", "NO_PROTON", "NO_XRS_OR_PROTON")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def seal(
    issue,
    *,
    full=0.8,
    no_xrs=0.7,
    no_proton=0.6,
    solar=0.4,
    thresholds=None,
    package_sha=SHA_A,
):
    thresholds = thresholds or {
        state: {"MAX_TSS": 0.5, "POD80_MIN_FAR": 0.4}
        for state in STATES
    }
    return build_forecast_seal(
        issued_at=issue,
        sealed_at=issue + timedelta(seconds=20),
        package_manifest_sha256=package_sha,
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
        architecture_id="fixture-v2",
    )


def five_minute_series(start, end, *, default=1.0, changes=None):
    times = []
    flux = []
    changes = dict(changes or {})
    current = start
    while current <= end:
        times.append(current)
        flux.append(float(changes.get(current, default)))
        current += timedelta(minutes=5)
    return times, flux


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
                architecture_id="fixture-v2",
            )

    def test_rehashed_invalid_probability_is_still_rejected_semantically(self):
        tampered = deepcopy(seal(BASE))
        tampered["probabilities"]["FULL"] = 1.5
        unsigned = dict(tampered)
        unsigned.pop("forecast_seal_sha256")
        tampered["forecast_seal_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
        with self.assertRaisesRegex(SealedEvaluationError, "outside"):
            validate_forecast_seal(tampered)

    def test_rehashed_wrong_horizon_is_rejected_semantically(self):
        tampered = deepcopy(seal(BASE))
        tampered["outcome_window_end"] = (BASE + timedelta(hours=23)).isoformat()
        unsigned = dict(tampered)
        unsigned.pop("forecast_seal_sha256")
        tampered["forecast_seal_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
        with self.assertRaisesRegex(SealedEvaluationError, "24 hours"):
            validate_forecast_seal(tampered)

    def test_two_sample_series_cannot_become_false_negative(self):
        s = seal(BASE)
        labels = derive_new_crossing_labels(
            forecast_seals=[s],
            proton_times=[BASE - timedelta(minutes=5), BASE + timedelta(minutes=5)],
            proton_flux=[1.0, 1.0],
        )
        row = labels["rows"][0]
        self.assertEqual(labels["format"], LABEL_FORMAT)
        self.assertFalse(row["outcome_resolved"])
        self.assertEqual(row["resolution_reason"], "OUTCOME_WINDOW_IMMATURE")
        self.assertIsNone(row["label"])

    def test_apparent_positive_with_large_gap_remains_unresolved(self):
        s = seal(BASE)
        labels = derive_new_crossing_labels(
            forecast_seals=[s],
            proton_times=[
                BASE - timedelta(minutes=5),
                BASE + timedelta(minutes=5),
                BASE + timedelta(hours=24),
            ],
            proton_flux=[1.0, 12.0, 12.0],
        )
        row = labels["rows"][0]
        self.assertFalse(row["outcome_resolved"])
        self.assertEqual(row["resolution_reason"], "OUTCOME_GAP_EXCEEDS_5_MINUTES")
        self.assertIsNone(row["label"])

    def test_complete_series_resolves_crossing_inside_24h(self):
        s = seal(BASE)
        cross = BASE + timedelta(hours=12)
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5),
            BASE + timedelta(hours=24),
            changes={cross: 12.0},
        )
        cross_index = times.index(cross)
        for i in range(cross_index, len(flux)):
            flux[i] = 12.0
        labels = derive_new_crossing_labels(
            forecast_seals=[s], proton_times=times, proton_flux=flux
        )
        row = labels["rows"][0]
        self.assertTrue(row["outcome_resolved"])
        self.assertTrue(row["eligible_new_crossing_issue"])
        self.assertEqual(row["label"], 1)
        self.assertEqual(row["first_crossing_utc"], cross.isoformat())
        self.assertLessEqual(row["max_gap_seconds"], 300.0)

    def test_crossing_after_horizon_does_not_label_positive(self):
        s = seal(BASE)
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5),
            BASE + timedelta(hours=24, minutes=5),
            default=1.0,
        )
        flux[-1] = 12.0
        labels = derive_new_crossing_labels(
            forecast_seals=[s], proton_times=times, proton_flux=flux
        )
        row = labels["rows"][0]
        self.assertTrue(row["outcome_resolved"])
        self.assertEqual(row["label"], 0)

    def test_duplicate_proton_timestamps_are_rejected(self):
        s = seal(BASE)
        with self.assertRaisesRegex(SealedEvaluationError, "duplicate proton"):
            derive_new_crossing_labels(
                forecast_seals=[s],
                proton_times=[BASE - timedelta(minutes=5), BASE - timedelta(minutes=5)],
                proton_flux=[1.0, 1.0],
            )

    def test_missing_exact_horizon_endpoint_is_unresolved(self):
        s = seal(BASE)
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5),
            BASE + timedelta(hours=24, minutes=5),
            default=1.0,
        )
        endpoint = BASE + timedelta(hours=24)
        keep = [i for i, value in enumerate(times) if value != endpoint]
        labels = derive_new_crossing_labels(
            forecast_seals=[s],
            proton_times=[times[i] for i in keep],
            proton_flux=[flux[i] for i in keep],
        )
        row = labels["rows"][0]
        self.assertFalse(row["outcome_resolved"])
        self.assertEqual(row["resolution_reason"], "OUTCOME_HORIZON_ENDPOINT_MISSING")

    def test_issue_already_above_threshold_is_resolved_but_ineligible(self):
        s = seal(BASE)
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5), BASE + timedelta(hours=24), default=11.0
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[s], proton_times=times, proton_flux=flux
        )
        row = labels["rows"][0]
        self.assertTrue(row["outcome_resolved"])
        self.assertFalse(row["eligible_new_crossing_issue"])
        self.assertIsNone(row["label"])

    def test_sample_support_never_becomes_independence_claim(self):
        forecasts = [seal(BASE + timedelta(days=i), full=0.9 if i == 0 else 0.1) for i in range(3)]
        start = BASE - timedelta(minutes=5)
        end = BASE + timedelta(days=2, hours=24)
        times, flux = five_minute_series(start, end, default=1.0)
        cross = BASE + timedelta(hours=1)
        cross_index = times.index(cross)
        # Return below threshold before the second issue so only the first row is positive.
        reset = BASE + timedelta(hours=2)
        reset_index = times.index(reset)
        for i in range(cross_index, reset_index):
            flux[i] = 12.0
        labels = derive_new_crossing_labels(
            forecast_seals=forecasts, proton_times=times, proton_flux=flux
        )
        result = evaluate_sealed_cohort(
            forecast_seals=forecasts,
            label_receipt=labels,
            minimum_positive_support=1,
        )
        self.assertEqual(result["positives"], 1)
        self.assertTrue(result["positive_support_gate_passed"])
        self.assertFalse(result["independence_verified"])
        self.assertIn("INDEPENDENCE_UNVERIFIED", result["claim_status"])

    def test_frozen_threshold_cannot_change_between_forecasts(self):
        first = seal(BASE)
        changed = {state: {"MAX_TSS": 0.5, "POD80_MIN_FAR": 0.4} for state in STATES}
        changed["FULL"] = {"MAX_TSS": 0.6, "POD80_MIN_FAR": 0.4}
        second = seal(BASE + timedelta(days=1), thresholds=changed)
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5), BASE + timedelta(days=2), default=1.0
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[first, second], proton_times=times, proton_flux=flux
        )
        with self.assertRaisesRegex(SealedEvaluationError, "threshold changed"):
            evaluate_sealed_cohort(
                forecast_seals=[first, second],
                label_receipt=labels,
                minimum_positive_support=1,
            )

    def test_model_package_change_cannot_mix_cohort(self):
        first = seal(BASE, package_sha=SHA_A)
        second = seal(BASE + timedelta(days=1), package_sha="e" * 64)
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5), BASE + timedelta(days=2), default=1.0
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[first, second], proton_times=times, proton_flux=flux
        )
        with self.assertRaisesRegex(SealedEvaluationError, "package changed"):
            evaluate_sealed_cohort(
                forecast_seals=[first, second], label_receipt=labels
            )

    def test_missing_label_row_fails_instead_of_shrinking_cohort(self):
        first = seal(BASE)
        second = seal(BASE + timedelta(days=1))
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5), BASE + timedelta(days=2), default=1.0
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[first], proton_times=times, proton_flux=flux
        )
        with self.assertRaisesRegex(SealedEvaluationError, "exactly match"):
            evaluate_sealed_cohort(
                forecast_seals=[first, second], label_receipt=labels
            )

    def test_old_v1_label_receipt_is_rejected(self):
        s = seal(BASE)
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5), BASE + timedelta(hours=24), default=1.0
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[s], proton_times=times, proton_flux=flux
        )
        old = deepcopy(labels)
        old["format"] = "IRIS_SEP_SEALED_NEW_CROSSING_LABELS_V1"
        unsigned = dict(old)
        unsigned.pop("label_receipt_sha256")
        old["label_receipt_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
        with self.assertRaisesRegex(SealedEvaluationError, "V2 required"):
            evaluate_sealed_cohort(forecast_seals=[s], label_receipt=old)

    def test_abstain_metrics_distinguish_threshold_crossings_from_alerts(self):
        s = seal(BASE, solar=0.9)
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5), BASE + timedelta(hours=24), default=1.0
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[s], proton_times=times, proton_flux=flux
        )
        result = evaluate_sealed_cohort(
            forecast_seals=[s], label_receipt=labels, minimum_positive_support=1
        )
        solar = result["states"]["NO_XRS_OR_PROTON"]
        self.assertEqual(solar["numerical_threshold_metrics"]["fp"], 1)
        self.assertEqual(solar["permission_filtered_alert_metrics"]["fp"], 0)
        self.assertFalse(solar["alerts_permitted_by_policy"])

    def test_review_enrichment_uses_actual_rounded_fraction(self):
        forecasts = [
            seal(BASE + timedelta(days=i), full=0.9 - 0.1 * i)
            for i in range(3)
        ]
        times, flux = five_minute_series(
            BASE - timedelta(minutes=5), BASE + timedelta(days=3), default=1.0
        )
        cross = BASE + timedelta(hours=1)
        cross_idx = times.index(cross)
        reset = BASE + timedelta(hours=2)
        reset_idx = times.index(reset)
        for i in range(cross_idx, reset_idx):
            flux[i] = 12.0
        labels = derive_new_crossing_labels(
            forecast_seals=forecasts, proton_times=times, proton_flux=flux
        )
        result = evaluate_sealed_cohort(
            forecast_seals=forecasts,
            label_receipt=labels,
            review_fraction=0.05,
            minimum_positive_support=1,
        )
        full = result["states"]["FULL"]
        self.assertEqual(full["review_rows"], 1)
        self.assertAlmostEqual(full["actual_review_fraction"], 1 / 3)
        self.assertAlmostEqual(full["review_enrichment_vs_random"], 3.0)


if __name__ == "__main__":
    unittest.main()
