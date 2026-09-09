"""Tuning-boundary tests for Luna B's baseline interface.

All records here are synthetic metadata only. No SEP-PRISM, SEPVAL, or outcome
file is read.
"""

from __future__ import annotations

import unittest

try:  # Support both package and direct pytest collection.
    from .baseline_interface import (
        BaselineContractError,
        BaselineRunSpec,
        assert_causal_feature_names,
        assert_no_locked_test_access,
        assert_unique_row_ids,
        validate_prediction_records,
    )
except ImportError:  # pragma: no cover - collection-mode compatibility
    from baseline_interface import (  # type: ignore
        BaselineContractError,
        BaselineRunSpec,
        assert_causal_feature_names,
        assert_no_locked_test_access,
        assert_unique_row_ids,
        validate_prediction_records,
    )


class BaselineInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"row_id": "toy-001", "partition": "train"},
            {"row_id": "toy-002", "partition": "validation_threshold"},
        ]
        self.predictions = [
            {
                "row_id": "toy-001",
                "issue_time_utc": "2026-01-01T00:00:00Z",
                "partition": "train",
                "model_name": "elastic_net_logistic_regression",
                "seed": 7,
                "p_sep_10mev_10pfu": 0.25,
                "threshold_used": 0.5,
            }
        ]

    def test_run_spec_is_tuning_only(self) -> None:
        BaselineRunSpec(
            model_name="xgboost", seed=7, feature_set="causal_prism"
        ).validate()
        with self.assertRaises(BaselineContractError):
            BaselineRunSpec(
                model_name="xgboost",
                seed=7,
                feature_set="causal_prism",
                phase="final_evaluation",
            ).validate()

    def test_tuning_rows_pass_without_locked_test(self) -> None:
        assert_no_locked_test_access(self.rows)
        assert_unique_row_ids(self.rows)

    def test_locked_test_is_rejected_even_without_a_label(self) -> None:
        with self.assertRaises(BaselineContractError):
            assert_no_locked_test_access(
                [{"row_id": "toy-test", "partition": "locked_test"}]
            )

    def test_prediction_frame_passes(self) -> None:
        validate_prediction_records(self.predictions)

    def test_prediction_frame_rejects_test_partition_and_outcome(self) -> None:
        test_record = dict(self.predictions[0])
        test_record["partition"] = "locked_test"
        with self.assertRaises(BaselineContractError):
            validate_prediction_records([test_record])

        outcome_record = dict(self.predictions[0])
        outcome_record["y_true"] = 0
        with self.assertRaises(BaselineContractError):
            validate_prediction_records([outcome_record])

    def test_prediction_frame_rejects_duplicate_row_and_seed(self) -> None:
        with self.assertRaises(BaselineContractError):
            validate_prediction_records(self.predictions * 2)

    def test_future_or_label_feature_names_are_rejected(self) -> None:
        assert_causal_feature_names(["SHARP_USFLUX", "flare_count_24h"])
        with self.assertRaises(BaselineContractError):
            assert_causal_feature_names(["Future_OSEP_label"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
