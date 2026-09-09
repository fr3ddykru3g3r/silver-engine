from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from iris_report.iris_sep.tools.prepare_sepnet_v1_dual_target_development import (
    GENERAL_TARGET,
    MAX_FLUX_TARGET,
    OPERATIONAL_TARGET,
    ROLES,
    prepare,
    sha256_file,
)


class DualTargetDevelopmentCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.output = root / "dual.csv"
        cls.manifest_path = root / "manifest.json"
        cls.manifest = prepare(cls.output, cls.manifest_path)
        cls.frame = pd.read_csv(cls.output, float_precision="round_trip")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_schema_and_hash_are_exact(self) -> None:
        self.assertEqual(self.manifest["feature_column_count"], 98)
        self.assertEqual(self.manifest["output_sha256"], sha256_file(self.output))
        self.assertEqual(
            list(self.frame.columns[5:8]),
            [GENERAL_TARGET, OPERATIONAL_TARGET, MAX_FLUX_TARGET],
        )
        feature_columns = self.manifest["feature_columns"]
        self.assertEqual(len(feature_columns), 98)
        self.assertFalse(any(column.lower().startswith("future_") for column in feature_columns))

    def test_operational_target_never_exceeds_general_target(self) -> None:
        self.assertTrue((self.frame[OPERATIONAL_TARGET] <= self.frame[GENERAL_TARGET]).all())
        self.assertTrue((self.frame[MAX_FLUX_TARGET] >= 0).all())
        for role in ROLES:
            role_frame = self.frame.loc[self.frame["role"] == role]
            self.assertEqual(role_frame[GENERAL_TARGET].nunique(), 2)
            self.assertEqual(role_frame[OPERATIONAL_TARGET].nunique(), 2)

    def test_roles_and_purge_match_frozen_v3_counts(self) -> None:
        expected = {
            "train": (7812, 1382, 1318),
            "validation_monitor": (1530, 244, 112),
            "validation_calibration": (1584, 329, 65),
            "validation_threshold": (776, 111, 192),
        }
        for role, (rows, units, operational_positives) in expected.items():
            actual = self.manifest["role_counts"][role]
            self.assertEqual(actual["rows"], rows)
            self.assertEqual(actual["units"], units)
            self.assertEqual(actual["operational_positive_windows"], operational_positives)
        self.assertEqual(self.manifest["purged_units"], 2)
        ordered = self.frame.assign(window_end=pd.to_datetime(self.frame["window_end"], utc=True))
        prior_end = None
        for role in ROLES:
            role_frame = ordered.loc[ordered["role"] == role]
            if prior_end is not None:
                self.assertGreater(role_frame["window_end"].min(), prior_end + pd.Timedelta(hours=24))
            prior_end = role_frame["window_end"].max()

    def test_manifest_is_serialized_and_outputs_are_immutable(self) -> None:
        self.assertEqual(json.loads(self.manifest_path.read_text()), self.manifest)
        with self.assertRaisesRegex(ValueError, "immutable"):
            prepare(self.output, self.manifest_path)


if __name__ == "__main__":
    unittest.main()
