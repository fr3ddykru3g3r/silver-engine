from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from iris_report.iris_sep.src.iris_sep.missingness_recovery import ReconstructionProvenance
from iris_report.iris_sep.src.iris_sep.operator_missing_data import (
    MissingDataResolutionPolicy,
    ObservedSourceRecord,
    resolve_modality_input,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
UTC = timezone.utc


class OperatorMissingDataTests(unittest.TestCase):
    def setUp(self):
        self.issue = datetime(2026, 9, 6, 12, tzinfo=UTC)
        self.policy = MissingDataResolutionPolicy(
            policy_id="MISSING_DATA_V1_TEST",
            maximum_observation_age_hours={"XRS": 3.0, "PROTON": 3.0},
            allowed_primary_sources={"XRS": ("GOES_XRS_PRIMARY",), "PROTON": ("GOES_PROTON_PRIMARY",)},
            allowed_alternate_sources={"XRS": ("GOES_XRS_BACKUP",), "PROTON": ("GOES_PROTON_BACKUP",)},
            allowed_source_revisions={"XRS": ("r1",), "PROTON": ("r1",)},
            allowed_reconstruction_method_ids={"XRS": ("CAUSAL_FF_V1",), "PROTON": ("CAUSAL_FF_V1",)},
            maximum_reconstruction_gap_hours={"XRS": 72.0, "PROTON": 24.0},
            maximum_reconstruction_uncertainty={"XRS": 0.30, "PROTON": 0.20},
            reconstruction_evidence_sha256={"XRS": SHA_C, "PROTON": SHA_D},
            structural_mask_supported_modalities=("XRS",),
            mask_aware_transient_supported_modalities=("XRS",),
        )

    def observed(self, source="GOES_XRS_PRIMARY", *, age=1.0, backup=False):
        return ObservedSourceRecord(
            source_id=source,
            source_revision="r1",
            observed_at=self.issue - timedelta(hours=age),
            published_at=self.issue - timedelta(minutes=10),
            payload_sha256=SHA_A,
            harmonization_evidence_sha256=SHA_B if backup else None,
        )

    def reconstruction(self, *, modality="XRS", uncertainty=0.1, latest_age=73.0, method="CAUSAL_FF_V1"):
        return ReconstructionProvenance(
            modality=modality,
            method_id=method,
            method_class="CAUSAL_STATISTICAL",
            fit_role="train",
            latest_observation_at=self.issue - timedelta(hours=latest_age),
            generated_at=self.issue,
            uses_future_information=False,
            physics_constraints=(),
            normalized_uncertainty=uncertainty,
            artifact_sha256=SHA_A,
        )

    def test_primary_observed_is_valid(self):
        result = resolve_modality_input(
            modality="XRS", missingness_class="OBSERVED", issued_at=self.issue,
            policy=self.policy, primary_observation=self.observed(),
        )
        self.assertEqual(result.value_origin, "PRIMARY_OBSERVED")
        self.assertEqual(result.forecast_permission, "VALID")
        self.assertFalse(result.reconstruction_is_observation)

    def test_valid_real_alternate_is_preferred_over_reconstruction(self):
        stale = self.observed(age=5.0)
        backup = self.observed(source="GOES_XRS_BACKUP", age=1.0, backup=True)
        result = resolve_modality_input(
            modality="XRS", missingness_class="TRANSIENT_MISSING", issued_at=self.issue,
            policy=self.policy, primary_observation=stale, alternate_observations=(backup,),
            reconstruction_provenance=self.reconstruction(), reconstruction_gap_hours=24,
            reconstruction_evidence_sha256=SHA_C,
        )
        self.assertEqual(result.value_origin, "ALTERNATE_OBSERVED")
        self.assertEqual(result.forecast_permission, "VALID")
        self.assertEqual(result.selected_source_id, "GOES_XRS_BACKUP")

    def test_alternate_without_harmonization_evidence_is_not_admitted(self):
        backup = self.observed(source="GOES_XRS_BACKUP", age=1.0, backup=False)
        result = resolve_modality_input(
            modality="XRS", missingness_class="TRANSIENT_MISSING", issued_at=self.issue,
            policy=self.policy, alternate_observations=(backup,),
        )
        self.assertEqual(result.value_origin, "NONE")
        self.assertEqual(result.forecast_permission, "DEGRADED")  # explicit mask-aware XRS fallback

    def test_valid_reconstruction_is_degraded_not_observed(self):
        result = resolve_modality_input(
            modality="XRS", missingness_class="TRANSIENT_MISSING", issued_at=self.issue,
            policy=self.policy, reconstruction_provenance=self.reconstruction(),
            reconstruction_gap_hours=24.0, reconstruction_evidence_sha256=SHA_C,
        )
        self.assertEqual(result.value_origin, "RECONSTRUCTED")
        self.assertEqual(result.forecast_permission, "DEGRADED")
        self.assertFalse(result.reconstruction_is_observation)

    def test_reconstruction_past_validated_horizon_fails_closed_without_mask_path(self):
        result = resolve_modality_input(
            modality="PROTON", missingness_class="TRANSIENT_MISSING", issued_at=self.issue,
            policy=self.policy, reconstruction_provenance=self.reconstruction(modality="PROTON", latest_age=49),
            reconstruction_gap_hours=48.0, reconstruction_evidence_sha256=SHA_D,
        )
        self.assertEqual(result.forecast_permission, "ABSTAIN")
        self.assertIn("RECONSTRUCTION_HORIZON_EXCEEDED", result.reasons)

    def test_reconstruction_evidence_must_match(self):
        result = resolve_modality_input(
            modality="PROTON", missingness_class="TRANSIENT_MISSING", issued_at=self.issue,
            policy=self.policy, reconstruction_provenance=self.reconstruction(modality="PROTON", latest_age=25),
            reconstruction_gap_hours=24.0, reconstruction_evidence_sha256=SHA_C,
        )
        self.assertEqual(result.forecast_permission, "ABSTAIN")
        self.assertIn("RECONSTRUCTION_EVIDENCE_MISMATCH", result.reasons)

    def test_high_reconstruction_uncertainty_fails_closed(self):
        result = resolve_modality_input(
            modality="PROTON", missingness_class="TRANSIENT_MISSING", issued_at=self.issue,
            policy=self.policy,
            reconstruction_provenance=self.reconstruction(modality="PROTON", uncertainty=0.9, latest_age=25),
            reconstruction_gap_hours=24.0, reconstruction_evidence_sha256=SHA_D,
        )
        self.assertEqual(result.forecast_permission, "ABSTAIN")
        self.assertIn("RECONSTRUCTION_UNCERTAINTY_TOO_HIGH", result.reasons)

    def test_future_information_reconstruction_fails_closed(self):
        bad = ReconstructionProvenance(
            modality="PROTON", method_id="CAUSAL_FF_V1", method_class="CAUSAL_STATISTICAL",
            fit_role="train", latest_observation_at=self.issue - timedelta(hours=25),
            generated_at=self.issue, uses_future_information=True, physics_constraints=(),
            normalized_uncertainty=0.1, artifact_sha256=SHA_A,
        )
        result = resolve_modality_input(
            modality="PROTON", missingness_class="TRANSIENT_MISSING", issued_at=self.issue,
            policy=self.policy, reconstruction_provenance=bad,
            reconstruction_gap_hours=24.0, reconstruction_evidence_sha256=SHA_D,
        )
        self.assertEqual(result.forecast_permission, "ABSTAIN")
        self.assertIn("FUTURE_INFORMATION_USED", result.reasons)

    def test_structural_unavailability_never_enters_reconstruction(self):
        result = resolve_modality_input(
            modality="XRS", missingness_class="STRUCTURAL_UNAVAILABLE", issued_at=self.issue,
            policy=self.policy, reconstruction_provenance=self.reconstruction(),
            reconstruction_gap_hours=24.0, reconstruction_evidence_sha256=SHA_C,
        )
        self.assertEqual(result.value_origin, "NONE")
        self.assertEqual(result.forecast_permission, "DEGRADED")
        self.assertIn("STRUCTURAL_UNAVAILABILITY_CANNOT_BE_RECONSTRUCTED_AS_OBSERVED", result.reasons)
        self.assertIn("STRUCTURAL_MASK_AWARE_PATH", result.reasons)

    def test_unsupported_structural_proton_abstains(self):
        result = resolve_modality_input(
            modality="PROTON", missingness_class="STRUCTURAL_UNAVAILABLE", issued_at=self.issue,
            policy=self.policy,
        )
        self.assertEqual(result.forecast_permission, "ABSTAIN")
        self.assertIn("STRUCTURAL_UNAVAILABLE_UNSUPPORTED", result.reasons)


if __name__ == "__main__":
    unittest.main()
