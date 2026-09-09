"""Dependency-light synthetic tests for hardened model semantics."""

from __future__ import annotations

import unittest

import numpy as np

from .luna_model import (
    BatchTargets,
    ModelConfig,
    discrete_time_censored_nll,
    ensemble_summary,
    event_conditional_peak_flux_loss,
    fuse_modalities,
    multitask_loss,
    prepare_feature_batch,
    sample_modality_keep_mask,
)


class ModelContractTests(unittest.TestCase):
    def test_feature_masks_preserve_which_feature_is_missing(self) -> None:
        values = {"magnetic": np.array([[[1.0, 99.0], [3.0, 4.0]]])}
        masks = {"magnetic": np.array([[[1, 0], [1, 1]]], dtype=bool)}
        batch = prepare_feature_batch(values, masks)
        np.testing.assert_array_equal(batch.feature_masks["magnetic"], masks["magnetic"])
        np.testing.assert_array_equal(batch.values["magnetic"][0, 0], [1.0, 0.0])

    def test_fusion_ignores_missing_and_flags_all_missing(self) -> None:
        representations = np.array([[[2.0, 4.0], [100.0, 100.0]], [[8.0, 8.0], [9.0, 9.0]]])
        observed = np.array([[True, False], [False, False]])
        fused, diagnostics = fuse_modalities(representations, observed, fallback=np.array([7.0, 9.0]))
        np.testing.assert_allclose(fused, [[2.0, 4.0], [7.0, 9.0]])
        self.assertEqual(diagnostics.all_missing.tolist(), [False, True])

    def test_dropout_fallback_never_invents_unavailable_feed(self) -> None:
        available = np.array([[True, True, False], [False, True, True], [False, False, False]])
        keep = sample_modality_keep_mask(available, 0.999, rng=np.random.default_rng(42))
        self.assertTrue(np.all(~keep | available))
        self.assertEqual(keep[:2].sum(axis=1).tolist(), [1, 1])
        self.assertEqual(int(keep[2].sum()), 0)

    def test_peak_loss_is_event_conditional(self) -> None:
        loss, denominator = event_conditional_peak_flux_loss(
            np.array([1.0, -1.0, 0.0, 8.0]),
            np.array([2.0, 9.0, 5.0, 100.0]),
            np.array([1, 0, 1, 1], dtype=bool),
            np.array([1, 1, 0, 0], dtype=bool),
        )
        self.assertEqual(denominator, 1)
        self.assertAlmostEqual(loss, 0.5)

    def test_event_hazard_does_not_pay_survival_after_event(self) -> None:
        logits = np.array([[2.0, 2.0, 100.0], [-2.0, -2.0, -2.0]])
        loss, denominator = discrete_time_censored_nll(
            logits, np.array([1, 3]), np.array([True, False])
        )
        event = np.logaddexp(0.0, 2.0) + np.logaddexp(0.0, -2.0)
        censored = 3.0 * np.logaddexp(0.0, -2.0)
        self.assertEqual(denominator, 2)
        self.assertAlmostEqual(loss, float((event + censored) / 2.0))

    def test_primary_only_default_and_masked_auxiliaries(self) -> None:
        targets = BatchTargets(
            occurrence=np.array([1, 0]),
            occurrence_valid=np.array([1, 1], dtype=bool),
            peak_log_flux=np.array([2.0, 99.0]),
            peak_valid=np.array([1, 1], dtype=bool),
            onset_bin=np.array([0, 2]),
            onset_event=np.array([1, 0], dtype=bool),
            onset_valid=np.array([1, 1], dtype=bool),
        )
        predictions = {
            "occurrence_logits": np.array([0.0, 0.0]),
            "peak_prediction": np.array([2.0, 2.0]),
            "onset_logits": np.zeros((2, 2)),
        }
        primary = multitask_loss(predictions, targets, ModelConfig())
        self.assertEqual(primary.active_tasks, ("occurrence",))
        expanded = multitask_loss(
            predictions,
            targets,
            ModelConfig(tasks=("occurrence", "peak", "onset"), weights={"peak": 2.0}),
        )
        self.assertEqual(expanded.denominators["peak"], 1)

    def test_ensemble_uses_median_and_abstains_when_all_missing(self) -> None:
        values = np.array([[0.1, 0.8], [0.2, 0.7], [0.9, 0.9]])
        summary = ensemble_summary(values, all_missing=np.array([False, True]), fallback_probability=0.25)
        np.testing.assert_allclose(summary.probability, [0.2, 0.25])
        self.assertEqual(summary.abstain.tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
