from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from iris_report.iris_sep.tools.prepare_sepnet_v1_dual_target_development_v5 import (
    EXPECTED_V3_CSV_SHA256,
    EXPECTED_V3_MANIFEST_SHA256,
    MAPPING_COLUMNS,
    PINNED_V3_CSV,
    TARGET_SCHEMA,
    _assert_strict_role_purge,
    _assert_v3_mapping,
    _assert_valid_max_flux,
    prepare,
)
from iris_report.iris_sep.tools.prepare_sepnet_v1_development import sha256_file


class DualTargetV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "cohort.csv"
        cls.manifest_path = cls.root / "receipt.json"
        cls.manifest = prepare(cls.output, cls.manifest_path)
        cls.frame = pd.read_csv(cls.output, float_precision="round_trip")
        cls.frozen = pd.read_csv(PINNED_V3_CSV, float_precision="round_trip")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_pinned_inputs_and_exact_ordered_mapping(self) -> None:
        self.assertEqual(sha256_file(PINNED_V3_CSV), EXPECTED_V3_CSV_SHA256)
        self.assertEqual(self.manifest["frozen_v3_manifest_sha256"], EXPECTED_V3_MANIFEST_SHA256)
        self.assertEqual(list(self.manifest["frozen_v3_mapping_columns"]), list(MAPPING_COLUMNS))
        mapping_hash = _assert_v3_mapping(self.frame, self.frozen)
        self.assertEqual(mapping_hash, self.manifest["frozen_v3_mapping_sha256"])

    def test_schema_and_all_provenance_hashes_are_bound(self) -> None:
        self.assertEqual(len(self.manifest["ordered_feature_columns"]), 98)
        self.assertEqual(self.manifest["ordered_target_schema"], list(TARGET_SCHEMA))
        self.assertEqual(
            set(self.manifest["source_code_sha256"]),
            {"v5_tool", "v4_dual_target_builder", "v3_prepare_and_build_units", "cohort_assignment_implementation"},
        )
        for digest in self.manifest["source_code_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(json.loads(self.manifest_path.read_text()), self.manifest)

    def test_mapping_mutations_fail_closed(self) -> None:
        for column, replacement in (
            ("issue_id", "mutated"),
            ("role", "validation_threshold"),
            ("unit_id", "mutated"),
            ("window_begin", "1999-01-01T00:00:00Z"),
            ("window_end", "1999-01-02T00:00:00Z"),
            ("future_Operational_SEP_label", 1 - int(self.frame.loc[0, "future_Operational_SEP_label"])),
        ):
            mutated = self.frame.copy()
            mutated.loc[0, column] = replacement
            with self.assertRaisesRegex(ValueError, "differs from frozen v3 mapping"):
                _assert_v3_mapping(mutated, self.frozen)

    def test_exact_purge_endpoint_is_rejected(self) -> None:
        synthetic = pd.DataFrame(
            {
                "role": ["train", "validation_monitor", "validation_calibration", "validation_threshold"],
                "window_end": [
                    "2020-01-01T00:00:00Z",
                    "2020-01-02T00:00:00Z",  # exactly +24 h: forbidden
                    "2020-01-04T00:00:00Z",
                    "2020-01-06T00:00:00Z",
                ],
            }
        )
        with self.assertRaisesRegex(ValueError, "strict inclusive 24-hour purge"):
            _assert_strict_role_purge(synthetic)

    def test_nonfinite_and_negative_flux_are_rejected(self) -> None:
        for invalid in (float("nan"), float("inf"), float("-inf"), -0.1):
            mutated = self.frame.copy()
            mutated.loc[0, "future_SEP_MaxFlux"] = invalid
            with self.assertRaisesRegex(ValueError, "finite|nonnegative"):
                _assert_valid_max_flux(mutated)

    def test_path_aliases_and_existing_outputs_are_rejected(self) -> None:
        alias = self.root / "same-destination"
        with self.assertRaisesRegex(ValueError, "alias"):
            prepare(alias, self.root / "." / "same-destination")
        with self.assertRaisesRegex(ValueError, "immutable"):
            prepare(self.output, self.root / "unused-manifest.json")

    def test_atomic_writer_left_no_staging_files(self) -> None:
        leftovers = [path.name for path in self.root.iterdir() if path.name.startswith(".")]
        self.assertEqual(leftovers, [])
        self.assertEqual(sha256_file(self.output), self.manifest["output_sha256"])


if __name__ == "__main__":
    unittest.main()
