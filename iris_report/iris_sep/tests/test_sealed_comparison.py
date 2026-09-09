from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from iris_report.iris_sep.src.iris_sep.promoted_model_package import ARCHITECTURE, TARGET
from iris_report.iris_sep.src.iris_sep.sealed_comparison import (
    SealedComparisonError,
    build_external_frozen_comparator,
    build_fit_prevalence_climatology,
    evaluate_sealed_comparators,
    seal_comparison,
    validate_sealed_comparison,
)
from iris_report.iris_sep.src.iris_sep.sealed_evaluation import (
    build_forecast_seal,
    derive_new_crossing_labels,
)


ISSUE = datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def forecast_seal(issue=ISSUE, *, full=0.2):
    probabilities = {"FULL": full, "NO_XRS": 0.18, "NO_PROTON": 0.16, "NO_XRS_OR_PROTON": 0.1}
    thresholds = {state: {"MAX_TSS": 0.25, "POD80_MIN_FAR": 0.15} for state in probabilities}
    permissions = {"FULL": "NORMAL", "NO_XRS": "DEGRADED", "NO_PROTON": "DEGRADED", "NO_XRS_OR_PROTON": "ABSTAIN"}
    return build_forecast_seal(
        issued_at=issue,
        sealed_at=issue + timedelta(seconds=30),
        package_manifest_sha256=SHA_A,
        feature_row_sha256=SHA_B,
        source_authentication_sha256=SHA_C,
        causal_feature_derivation_sha256=SHA_D,
        probabilities=probabilities,
        thresholds=thresholds,
        operator_permissions=permissions,
        architecture_id=ARCHITECTURE,
    )


def manifest():
    return {
        "architecture": ARCHITECTURE,
        "target": TARGET,
        "fit_prevalence": 0.03125,
        "runtime_training_allowed": False,
    }


def full_outcome_series(start, end, *, crossing=None):
    times = []
    flux = []
    current = start
    crossed = False
    while current <= end:
        if crossing is not None and current >= crossing:
            crossed = True
        times.append(current)
        flux.append(12.0 if crossed else 1.0)
        current += timedelta(minutes=5)
    return times, flux


class SealedComparisonTests(unittest.TestCase):
    def test_fit_prevalence_is_frozen_from_package(self):
        comparator = build_fit_prevalence_climatology(
            package_manifest=manifest(),
            package_manifest_sha256=SHA_A,
        )
        self.assertEqual(comparator["probability"], 0.03125)
        self.assertFalse(comparator["trained_or_tuned_on_evaluation_cohort"])
        receipt = seal_comparison(
            forecast_seal=forecast_seal(),
            comparators=[comparator],
            sealed_at=ISSUE + timedelta(minutes=1),
        )
        self.assertEqual(validate_sealed_comparison(receipt), receipt)

    def test_external_comparator_requires_immutable_artifact(self):
        with self.assertRaisesRegex(SealedComparisonError, "SHA-256"):
            build_external_frozen_comparator(
                comparator_id="EXTERNAL",
                comparator_version="v1",
                artifact_sha256="not-a-sha",
                probability=0.2,
            )

    def test_comparison_sealed_too_late_is_blocked(self):
        comparator = build_fit_prevalence_climatology(
            package_manifest=manifest(), package_manifest_sha256=SHA_A
        )
        with self.assertRaisesRegex(SealedComparisonError, "five minutes"):
            seal_comparison(
                forecast_seal=forecast_seal(),
                comparators=[comparator],
                sealed_at=ISSUE + timedelta(minutes=6),
            )

    def test_post_outcome_tuned_comparator_is_blocked(self):
        comparator = build_external_frozen_comparator(
            comparator_id="EXTERNAL",
            comparator_version="v1",
            artifact_sha256=SHA_B,
            probability=0.2,
        )
        comparator["trained_or_tuned_on_evaluation_cohort"] = True
        with self.assertRaisesRegex(SealedComparisonError, "trained or tuned"):
            seal_comparison(
                forecast_seal=forecast_seal(),
                comparators=[comparator],
                sealed_at=ISSUE + timedelta(minutes=1),
            )

    def test_probability_tamper_breaks_seal_even_after_valid_creation(self):
        comparator = build_fit_prevalence_climatology(
            package_manifest=manifest(), package_manifest_sha256=SHA_A
        )
        receipt = seal_comparison(
            forecast_seal=forecast_seal(),
            comparators=[comparator],
            sealed_at=ISSUE + timedelta(minutes=1),
        )
        tampered = deepcopy(receipt)
        tampered["comparators"][0]["probability"] = 0.9
        with self.assertRaisesRegex(SealedComparisonError, "digest mismatch"):
            validate_sealed_comparison(tampered)

    def test_evaluation_uses_full_probability_from_sealed_forecast(self):
        forecast = forecast_seal(full=0.9)
        comparator = build_external_frozen_comparator(
            comparator_id="EXTERNAL",
            comparator_version="v1",
            artifact_sha256=SHA_B,
            probability=0.1,
        )
        comparison = seal_comparison(
            forecast_seal=forecast,
            comparators=[comparator],
            sealed_at=ISSUE + timedelta(minutes=1),
        )
        times, flux = full_outcome_series(
            ISSUE - timedelta(minutes=5),
            ISSUE + timedelta(hours=24),
            crossing=ISSUE + timedelta(hours=1),
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[forecast], proton_times=times, proton_flux=flux
        )
        result = evaluate_sealed_comparators(
            forecast_seals=[forecast],
            comparison_receipts=[comparison],
            label_receipt=labels,
        )
        row = result["comparators"]["EXTERNAL"]
        self.assertAlmostEqual(row["iris_full_brier"], 0.01)
        self.assertAlmostEqual(row["comparator_brier"], 0.81)
        self.assertTrue(result["iris_probabilities_sourced_from_forecast_seals"])
        self.assertTrue(result["common_cohort_verified"])

    def test_every_forecast_requires_comparison_receipt(self):
        first = forecast_seal()
        second = forecast_seal(ISSUE + timedelta(days=1))
        comparator = build_fit_prevalence_climatology(
            package_manifest=manifest(), package_manifest_sha256=SHA_A
        )
        comparison = seal_comparison(
            forecast_seal=first,
            comparators=[comparator],
            sealed_at=ISSUE + timedelta(minutes=1),
        )
        times, flux = full_outcome_series(
            ISSUE - timedelta(minutes=5), ISSUE + timedelta(days=2)
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[first, second], proton_times=times, proton_flux=flux
        )
        with self.assertRaisesRegex(SealedComparisonError, "exactly one"):
            evaluate_sealed_comparators(
                forecast_seals=[first, second],
                comparison_receipts=[comparison],
                label_receipt=labels,
            )

    def test_comparator_set_cannot_change_within_cohort(self):
        first = forecast_seal()
        second_issue = ISSUE + timedelta(days=1)
        second = forecast_seal(second_issue)
        climate = build_fit_prevalence_climatology(
            package_manifest=manifest(), package_manifest_sha256=SHA_A
        )
        external = build_external_frozen_comparator(
            comparator_id="EXTERNAL",
            comparator_version="v1",
            artifact_sha256=SHA_B,
            probability=0.2,
        )
        first_comp = seal_comparison(
            forecast_seal=first,
            comparators=[climate],
            sealed_at=ISSUE + timedelta(minutes=1),
        )
        second_comp = seal_comparison(
            forecast_seal=second,
            comparators=[climate, external],
            sealed_at=second_issue + timedelta(minutes=1),
        )
        times, flux = full_outcome_series(
            ISSUE - timedelta(minutes=5), ISSUE + timedelta(days=2)
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[first, second], proton_times=times, proton_flux=flux
        )
        with self.assertRaisesRegex(SealedComparisonError, "set/version changed"):
            evaluate_sealed_comparators(
                forecast_seals=[first, second],
                comparison_receipts=[first_comp, second_comp],
                label_receipt=labels,
            )

    def test_duplicate_comparison_receipt_cannot_inflate_support(self):
        forecast = forecast_seal()
        comparator = build_fit_prevalence_climatology(
            package_manifest=manifest(), package_manifest_sha256=SHA_A
        )
        comparison = seal_comparison(
            forecast_seal=forecast,
            comparators=[comparator],
            sealed_at=ISSUE + timedelta(minutes=1),
        )
        times, flux = full_outcome_series(
            ISSUE - timedelta(minutes=5), ISSUE + timedelta(hours=24)
        )
        labels = derive_new_crossing_labels(
            forecast_seals=[forecast], proton_times=times, proton_flux=flux
        )
        with self.assertRaisesRegex(SealedComparisonError, "duplicate comparison"):
            evaluate_sealed_comparators(
                forecast_seals=[forecast],
                comparison_receipts=[comparison, comparison],
                label_receipt=labels,
            )


if __name__ == "__main__":
    unittest.main()
