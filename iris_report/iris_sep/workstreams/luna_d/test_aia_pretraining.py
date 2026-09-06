"""Dependency-free synthetic tests for the isolated AIA pretraining interface."""

from __future__ import annotations

import unittest

from .aia_pretraining import (
    AIAContractError,
    build_auxiliary_pretraining_interface,
    build_pretraining_receipt,
    default_pretraining_spec,
    validate_pretraining_records,
    validate_pretraining_spec,
)


def _provenance() -> dict[str, str]:
    return {
        "source_uri": "synthetic://aarp-feature-slice",
        "source_version": "toy-v1",
        "source_sha256": "a" * 64,
        "source_row_id": "aia-row-1",
        "retrieved_at_utc": "2026-09-04T00:00:00Z",
    }


def _record() -> dict:
    sequence = [[float(row * 176 + column) for column in range(176)] for row in range(7)]
    mask = [[1 if (row + column) % 5 else 0 for column in range(176)] for row in range(7)]
    target = [
        [None if mask[row][column] else sequence[row][column] for column in range(176)]
        for row in range(7)
    ]
    return {
        "example_id": "aia-example-1",
        "aia_source_id": "2014.06.14_15:48:00_7h@1h_4228.fits",
        "issue_time_utc": "2014-06-14T15:48:00Z",
        "partition": "aia_pretraining_train",
        "sequence": sequence,
        "observed_mask": mask,
        "pretraining_target_kind": "masked_feature_reconstruction",
        "pretraining_target": target,
        "provenance": _provenance(),
    }


class AIApretrainingTests(unittest.TestCase):
    def test_default_spec_is_label_free_and_archive_download_is_prohibited(self) -> None:
        spec = validate_pretraining_spec(default_pretraining_spec())
        self.assertFalse(spec["sep_labels_used"])
        self.assertFalse(spec["locked_test_accessed"])
        self.assertEqual(spec["archive_policy"]["raw_archive_download"], "PROHIBITED")
        self.assertEqual(spec["target_namespace"], "aia_self_supervised")

    def test_valid_small_feature_batch_passes(self) -> None:
        validation = validate_pretraining_records([_record()])
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(validation["example_count"], 1)
        self.assertFalse(validation["sep_labels_used"])
        self.assertFalse(validation["raw_aia_archive_downloaded"])
        receipt = build_pretraining_receipt(validation, generated_at_utc="2026-09-04T01:00:00Z")
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(len(receipt["receipt_sha256"]), 64)

    def test_sep_labels_and_hmi_keys_are_rejected(self) -> None:
        labelled = _record()
        labelled["Future_OSEP_label"] = 0
        with self.assertRaises(AIAContractError):
            validate_pretraining_records([labelled])
        hmi = _record()
        hmi["harpnum"] = 501
        with self.assertRaises(AIAContractError):
            validate_pretraining_records([hmi])

    def test_locked_test_and_observed_target_leak_are_rejected(self) -> None:
        locked = _record()
        locked["partition"] = "locked_test"
        with self.assertRaises(AIAContractError):
            validate_pretraining_records([locked])
        leaked = _record()
        leaked["pretraining_target"][0][1] = leaked["sequence"][0][1]
        with self.assertRaises(AIAContractError):
            validate_pretraining_records([leaked])

    def test_shape_and_duplicate_identity_are_rejected(self) -> None:
        wrong_shape = _record()
        wrong_shape["sequence"] = wrong_shape["sequence"][:-1]
        with self.assertRaises(AIAContractError):
            validate_pretraining_records([wrong_shape])
        duplicate = _record()
        with self.assertRaises(AIAContractError):
            validate_pretraining_records([_record(), duplicate])

    def test_interface_declares_encoder_only_transfer(self) -> None:
        interface = build_auxiliary_pretraining_interface()
        rejected = interface["runner_requirements"]["rejected_inputs"]
        self.assertIn("SEP labels", rejected)
        self.assertIn("locked_test rows or outcomes", rejected)
        self.assertIn("HMI keys before the identity bridge", rejected)
        not_transferable = interface["spec"]["transfer_contract"]["not_transferable"]
        self.assertIn("sep_prediction_head", not_transferable)


if __name__ == "__main__":
    unittest.main()
