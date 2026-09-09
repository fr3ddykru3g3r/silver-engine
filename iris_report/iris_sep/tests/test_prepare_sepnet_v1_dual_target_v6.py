from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from iris_report.iris_sep.tools import prepare_sepnet_v1_dual_target_development as v4
from iris_report.iris_sep.tools import prepare_sepnet_v1_dual_target_development_v5 as v5
from iris_report.iris_sep.tools import prepare_sepnet_v1_dual_target_development_v6 as v6


class DualTargetV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.output = cls.root / "v6.csv"
        cls.receipt = cls.root / "v6.json"
        cls.manifest = v6.prepare(cls.output, cls.receipt)
        cls.raw = pd.read_csv(v4.PINNED_SOURCE)
        cls.candidate = pd.read_csv(cls.output, float_precision="round_trip")
        cls.frozen = pd.read_csv(v5.PINNED_V3_CSV, float_precision="round_trip")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_wrong_source_hash_fails(self) -> None:
        staged = self.root / "wrong-hash.csv"
        staged.write_bytes(v4.PINNED_SOURCE.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            v6._read_source_with_hash(staged, v4.EXPECTED_SOURCE_SHA256)

    def test_missing_each_required_target_fails(self) -> None:
        for column in (*v4.TARGET_COLUMNS, "window_begin", "window_end"):
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, "missing required"):
                    v6._validate_source_frame(self.raw.drop(columns=[column]))

    def test_duplicate_non24h_and_bad_cadence_fail(self) -> None:
        duplicate = self.raw.copy()
        duplicate.loc[1, ["window_begin", "window_end"]] = duplicate.loc[0, ["window_begin", "window_end"]].values
        with self.assertRaisesRegex(ValueError, "duplicate"):
            v6._validate_source_frame(duplicate)
        non24 = self.raw.copy()
        non24.loc[0, "window_end"] = str(pd.Timestamp(non24.loc[0, "window_end"]) + pd.Timedelta(hours=1))
        with self.assertRaisesRegex(ValueError, "exactly 24"):
            v6._validate_source_frame(non24)
        cadence = self.raw.copy()
        cadence.loc[1, "window_begin"] = str(pd.Timestamp(cadence.loc[1, "window_begin"]) + pd.Timedelta(hours=1))
        cadence.loc[1, "window_end"] = str(pd.Timestamp(cadence.loc[1, "window_end"]) + pd.Timedelta(hours=1))
        with self.assertRaisesRegex(ValueError, "daily cadence"):
            v6._validate_source_frame(cadence)

    def test_null_nonbinary_and_inconsistent_labels_fail(self) -> None:
        for target in (v4.GENERAL_TARGET, v4.OPERATIONAL_TARGET):
            for invalid in (float("nan"), 2, -1, 0.5):
                mutated = self.raw.copy()
                mutated[target] = mutated[target].astype(float)
                mutated.loc[0, target] = invalid
                with self.subTest(target=target, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "complete and binary"):
                        v6._validate_source_frame(mutated)
        inconsistent = self.raw.copy()
        index = inconsistent.index[inconsistent[v4.GENERAL_TARGET] == 0][0]
        inconsistent.loc[index, v4.OPERATIONAL_TARGET] = 1
        with self.assertRaisesRegex(ValueError, "subset"):
            v6._validate_source_frame(inconsistent)

    def test_missing_and_extra_predictor_fail(self) -> None:
        _, features = v6._validate_source_frame(self.raw)
        with self.assertRaisesRegex(ValueError, "98-column schema"):
            v6._validate_source_frame(self.raw.drop(columns=[features[0]]))
        extra = self.raw.copy()
        extra["unexpected_causal_predictor"] = 0.0
        with self.assertRaisesRegex(ValueError, "98-column schema"):
            v6._validate_source_frame(extra)

    def test_empty_and_single_class_roles_fail(self) -> None:
        empty = self.candidate.loc[self.candidate["role"] != "validation_monitor"].copy()
        with self.assertRaisesRegex(ValueError, "is empty"):
            v6._assert_role_classes(empty)
        single = self.candidate.copy()
        mask = single["role"] == "validation_monitor"
        single.loc[mask, v4.OPERATIONAL_TARGET] = 0
        with self.assertRaisesRegex(ValueError, "lacks both classes"):
            v6._assert_role_classes(single)

    def test_mapping_mutation_and_exact_boundary_fail(self) -> None:
        mutated = self.candidate.copy()
        mutated.loc[0, "unit_id"] = "mutation"
        with self.assertRaisesRegex(ValueError, "differs from frozen v3 mapping"):
            v6._assert_candidate(mutated, self.frozen)
        exact = pd.DataFrame({
            "role": list(v4.ROLES),
            "window_end": ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", "2020-01-04T00:00:00Z", "2020-01-06T00:00:00Z"],
        })
        with self.assertRaisesRegex(ValueError, "strict inclusive"):
            v5._assert_strict_role_purge(exact)

    def test_manifest_recomputes_all_hashes_and_forbids_claims(self) -> None:
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(), self.manifest["output_sha256"])
        self.assertEqual(v6._json_hash(self.manifest["ordered_feature_columns"]), self.manifest["ordered_feature_schema_sha256"])
        self.assertEqual(v6._json_hash(v6.TARGET_SCHEMA), self.manifest["ordered_target_schema_sha256"])
        self.assertEqual(v5._assert_v3_mapping(self.candidate, self.frozen), self.manifest["frozen_v3_mapping_sha256"])
        dependencies = {
            "v6_tool": Path(v6.__file__), "v5_tool": Path(v5.__file__),
            "v4_dual_target_builder": Path(v4.__file__),
            "v3_prepare_and_build_units": Path(v6.v3_builder.__file__),
            "cohort_assignment_implementation": Path(v6.cohort.__file__),
        }
        self.assertEqual(
            {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in dependencies.items()},
            self.manifest["source_code_sha256"],
        )
        self.assertTrue({"superiority", "production_readiness"}.issubset(self.manifest["forbidden_claims"]))
        self.assertFalse(self.manifest["testing_or_sepval_artifact_accessed"])

    def test_atomic_pair_failure_removes_only_own_output(self) -> None:
        output = self.root / "pair-failure.csv"
        receipt = self.root / "pair-failure.json"
        actual = v6._atomic_exclusive_publish
        calls = 0
        def fail_second(path: Path, payload: bytes):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FileExistsError("simulated manifest race")
            return actual(path, payload)
        with mock.patch.object(v6, "_atomic_exclusive_publish", side_effect=fail_second):
            with self.assertRaises(FileExistsError):
                v6.prepare(output, receipt)
        self.assertFalse(output.exists())
        self.assertFalse(receipt.exists())

    def test_inode_or_hash_change_prevents_cleanup(self) -> None:
        path = self.root / "owned-publication"
        identity = v6._atomic_exclusive_publish(path, b"ours")
        path.unlink()
        path.write_bytes(b"other invocation")
        self.assertFalse(v6._remove_only_own_publication(path, identity))
        self.assertEqual(path.read_bytes(), b"other invocation")
        path.unlink()
        identity = v6._atomic_exclusive_publish(path, b"ours-again")
        path.write_bytes(b"mutated-in-place")
        self.assertFalse(v6._remove_only_own_publication(path, identity))
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
