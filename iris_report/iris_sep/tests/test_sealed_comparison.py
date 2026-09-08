from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from iris_report.iris_sep.src.iris_sep.promoted_model_package import ARCHITECTURE, TARGET
from iris_report.iris_sep.src.iris_sep.sealed_comparison import (
    SealedComparisonError,
    build_external_frozen_comparator,
    build_fit_prevalence_climatology,
    seal_comparison,
    validate_sealed_comparison,
)
from iris_report.iris_sep.src.iris_sep.sealed_evaluation import build_forecast_seal


ISSUE = datetime(2026, 9, 8, 0, 0, tzinfo=timezone.utc)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def forecast_seal():
    probabilities = {"FULL": 0.2, "NO_XRS": 0.18, "NO_PROTON": 0.16, "NO_XRS_OR_PROTON": 0.1}
    thresholds = {state: {"MAX_TSS": 0.25, "POD80_MIN_FAR": 0.15} for state in probabilities}
    permissions = {"FULL": "NORMAL", "NO_XRS": "DEGRADED", "NO_PROTON": "DEGRADED", "NO_XRS_OR_PROTON": "ABSTAIN"}
    return build_forecast_seal(
        issued_at=ISSUE,
        sealed_at=ISSUE + timedelta(seconds=30),
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
            package_manifest=manifest(),
            package_manifest_sha256=SHA_A,
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
            package_manifest=manifest(),
            package_manifest_sha256=SHA_A,
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


if __name__ == "__main__":
    unittest.main()
