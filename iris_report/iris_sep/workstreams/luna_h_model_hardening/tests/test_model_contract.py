from __future__ import annotations

import numpy as np
import pytest

from luna_model import (
    BatchTargets,
    ModelConfig,
    ensemble_summary,
    event_conditional_peak_flux_loss,
    fuse_modalities,
    multitask_loss,
    prepare_feature_batch,
    discrete_time_censored_nll,
)


def test_feature_masks_are_preserved_and_modality_mask_is_derived() -> None:
    values = {"magnetic": np.array([[[1.0, 99.0], [3.0, 4.0]]])}
    masks = {"magnetic": np.array([[[1, 0], [1, 1]]], dtype=bool)}

    batch = prepare_feature_batch(values, masks)

    np.testing.assert_array_equal(batch.feature_masks["magnetic"], masks["magnetic"])
    np.testing.assert_array_equal(batch.values["magnetic"][0, 0], [1.0, 0.0])
    assert batch.modality_masks["magnetic"].tolist() == [True]
    assert batch.all_missing.tolist() == [False]


def test_missing_modality_fusion_averages_only_observed_modalities() -> None:
    representations = np.array([[[2.0, 4.0], [100.0, 100.0]]])
    observed = np.array([[True, False]])

    fused, diagnostics = fuse_modalities(representations, observed, fallback=np.array([7.0, 9.0]))

    np.testing.assert_allclose(fused, [[2.0, 4.0]])
    assert diagnostics.all_missing.tolist() == [False]
    assert diagnostics.observed_count.tolist() == [1]


def test_all_missing_uses_explicit_fallback_and_abstention_signal() -> None:
    representations = np.zeros((2, 3, 4))
    observed = np.array([[False, False, False], [True, False, False]])
    fused, diagnostics = fuse_modalities(representations, observed, fallback=np.ones(4))

    np.testing.assert_allclose(fused[0], np.ones(4))
    assert diagnostics.all_missing.tolist() == [True, False]

    summary = ensemble_summary(
        np.array([[0.1, 0.8], [0.2, 0.7], [0.3, 0.9]]),
        all_missing=diagnostics.all_missing,
        fallback_probability=0.25,
    )
    np.testing.assert_allclose(summary.probability, [0.25, 0.8])
    assert summary.abstain.tolist() == [True, False]
    assert summary.epistemic_std[0] == pytest.approx(np.std([0.1, 0.2, 0.3]))


def test_peak_flux_loss_is_conditional_on_valid_events() -> None:
    logits = np.array([1.0, -1.0, 0.0, 8.0])
    peak = np.array([2.0, 9.0, 5.0, 100.0])
    occurrence = np.array([1, 0, 1, 1], dtype=bool)
    valid = np.array([1, 1, 0, 0], dtype=bool)

    loss, denominator = event_conditional_peak_flux_loss(logits, peak, occurrence, valid)

    assert denominator == 1
    assert loss == pytest.approx(0.5)


def test_censored_survival_nll_distinguishes_event_bin_and_censoring() -> None:
    # Three bins. Event at bin 1: survive bin 0, then fail at bin 1.
    logits = np.array([[2.0, 2.0, 2.0], [-2.0, -2.0, -2.0]])
    onset_bin = np.array([1, 3])
    observed_event = np.array([True, False])

    loss, denominator = discrete_time_censored_nll(logits, onset_bin, observed_event)

    expected_event = -(
        np.log(1.0 - 1.0 / (1.0 + np.exp(-2.0)))
        + np.log(1.0 / (1.0 + np.exp(-2.0)))
    )
    expected_censored = -3.0 * np.log(1.0 - 1.0 / (1.0 + np.exp(2.0)))
    assert denominator == 2
    assert loss == pytest.approx((expected_event + expected_censored) / 2.0)


def test_primary_only_is_default_and_auxiliary_weights_are_configurable() -> None:
    config = ModelConfig()
    assert config.active_tasks == ("occurrence",)

    targets = BatchTargets(
        occurrence=np.array([1, 0]),
        occurrence_valid=np.array([1, 1], dtype=bool),
        peak_log_flux=np.array([2.0, 99.0]),
        peak_valid=np.array([1, 1], dtype=bool),
        onset_bin=np.array([0, 2]),
        onset_event=np.array([1, 0], dtype=bool),
        onset_valid=np.array([1, 1], dtype=bool),
    )
    predictions = {"occurrence_logits": np.array([0.0, 0.0]), "peak_prediction": np.array([2.0, 2.0]), "onset_logits": np.zeros((2, 2))}
    primary = multitask_loss(predictions, targets, config)
    assert primary.active_tasks == ("occurrence",)
    assert set(primary.per_task) == {"occurrence"}

    expanded = multitask_loss(predictions, targets, ModelConfig(tasks=("occurrence", "peak", "onset"), weights={"peak": 2.0, "onset": 0.5}))
    assert expanded.active_tasks == ("occurrence", "peak", "onset")
    assert expanded.denominators["peak"] == 1
    assert expanded.denominators["onset"] == 2


def test_invalid_task_configuration_fails_closed() -> None:
    with pytest.raises(ValueError, match="occurrence"):
        ModelConfig(tasks=("peak",))
