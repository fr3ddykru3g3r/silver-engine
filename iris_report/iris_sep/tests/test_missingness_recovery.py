import unittest
from datetime import datetime, timezone

import numpy as np

from iris_report.iris_sep.src.iris_sep.missingness_recovery import (
    ReconstructionProvenance,
    audit_forecast_time_reconstruction,
    contiguous_gap_mask,
    deterministic_random_gap_mask,
    interval_coverage_metrics,
    reconstruction_metrics,
    reconstruction_payload_sha256,
)


UTC = timezone.utc
ISSUE = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
HASH = "a" * 64


def provenance(**changes):
    values = dict(
        modality="magnetic",
        method_id="sft-pfss-v1",
        method_class="PHYSICS_ASSIMILATED",
        fit_role="train",
        latest_observation_at=datetime(2026, 9, 5, 11, 30, tzinfo=UTC),
        generated_at=datetime(2026, 9, 5, 11, 45, tzinfo=UTC),
        uses_future_information=False,
        physics_constraints=("surface_flux_transport", "divergence_control"),
        normalized_uncertainty=0.2,
        artifact_sha256=HASH,
    )
    values.update(changes)
    return ReconstructionProvenance(**values)


class MissingnessRecoveryTests(unittest.TestCase):
    def test_causal_declared_physics_reconstruction_passes_provenance_gate(self):
        reasons = audit_forecast_time_reconstruction(
            provenance(), issued_at=ISSUE,
            allowed_method_ids={"magnetic": ("sft-pfss-v1",)},
            maximum_uncertainty=0.4,
            require_declared_physics=True,
        )
        self.assertEqual(reasons, ())

    def test_future_unapproved_uncertain_reconstruction_fails_closed(self):
        item = provenance(
            method_id="unfrozen",
            latest_observation_at=datetime(2026, 9, 5, 12, 1, tzinfo=UTC),
            generated_at=datetime(2026, 9, 5, 12, 2, tzinfo=UTC),
            uses_future_information=True,
            normalized_uncertainty=0.9,
        )
        reasons = audit_forecast_time_reconstruction(
            item, issued_at=ISSUE,
            allowed_method_ids={"magnetic": ("sft-pfss-v1",)},
            maximum_uncertainty=0.4,
        )
        self.assertEqual(set(reasons), {
            "FUTURE_INFORMATION_USED",
            "OBSERVATION_AFTER_ISSUE_TIME",
            "RECONSTRUCTION_GENERATED_AFTER_ISSUE_TIME",
            "RECONSTRUCTION_METHOD_NOT_ALLOWED",
            "RECONSTRUCTION_UNCERTAINTY_TOO_HIGH",
        })

    def test_physics_method_requires_declared_constraints_when_requested(self):
        reasons = audit_forecast_time_reconstruction(
            provenance(physics_constraints=()), issued_at=ISSUE,
            allowed_method_ids={"magnetic": ("sft-pfss-v1",)},
            maximum_uncertainty=0.4,
            require_declared_physics=True,
        )
        self.assertEqual(reasons, ("PHYSICS_CONSTRAINTS_UNDECLARED",))

    def test_random_gap_mask_is_deterministic_and_nontrivial(self):
        first = deterministic_random_gap_mask((10, 4), 0.25, seed=7)
        second = deterministic_random_gap_mask((10, 4), 0.25, seed=7)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(int(first.sum()), 10)
        self.assertFalse(first.all())

    def test_contiguous_gap_mask_builds_outage_block(self):
        mask = contiguous_gap_mask((8, 3), axis=0, start=2, length=3)
        self.assertTrue(mask[2:5].all())
        self.assertFalse(mask[:2].any())
        self.assertFalse(mask[5:].any())

    def test_reconstruction_metrics_score_only_hidden_cells(self):
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        reconstruction = np.array([9999.0, 1.0, 9999.0, 6.0])
        mask = np.array([False, True, False, True])
        metrics = reconstruction_metrics(truth, reconstruction, mask)
        self.assertEqual(metrics["held_out_count"], 2)
        self.assertAlmostEqual(metrics["mae"], 1.5)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(metrics["bias"], 0.5)

    def test_interval_coverage_is_measured_only_on_hidden_cells(self):
        truth = np.array([100.0, 2.0, 3.0])
        reconstruction = np.array([-100.0, 2.1, 5.0])
        sigma = np.array([0.0, 0.1, 0.25])
        mask = np.array([False, True, True])
        metrics = interval_coverage_metrics(truth, reconstruction, sigma, mask, z=2.0)
        self.assertEqual(metrics["held_out_count"], 2)
        self.assertAlmostEqual(metrics["interval_coverage"], 0.5)

    def test_nonfinite_hidden_reconstruction_is_rejected(self):
        with self.assertRaises(ValueError):
            reconstruction_metrics([1.0, 2.0], [1.0, float("nan")], [False, True])

    def test_payload_hash_binds_exact_reconstruction_values(self):
        first = reconstruction_payload_sha256([1.0, 2.0, 3.0])
        second = reconstruction_payload_sha256([1.0, 2.0, 3.0])
        third = reconstruction_payload_sha256([1.0, 2.0, 3.000001])
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)


if __name__ == "__main__":
    unittest.main()
