"""Synthetic-only tests for the fail-closed cohort pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from .iris_sep_pipeline import (
    CohortUnit,
    FeatureRecord,
    FeatureRow,
    Observation,
    TrainOnlyStandardizer,
    assign_chronological_roles,
    build_cohort_units,
    build_targets,
    freeze_manifest,
    make_issue,
    validate_features,
    verify_manifest,
    write_immutable_manifest,
)
from .iris_sep_pipeline.errors import PipelineError, ProtectedDataError


UTC = timezone.utc
BASE = datetime(2026, 1, 1, tzinfo=UTC)


class PipelineTests(unittest.TestCase):
    def test_feature_must_be_published_by_issue_time(self) -> None:
        issue = make_issue("synthetic", BASE)
        invalid = FeatureRecord(
            issue.issue_id, "x", BASE - timedelta(hours=2), BASE + timedelta(seconds=1), 1.0
        )
        with self.assertRaisesRegex(PipelineError, "published after"):
            validate_features([issue], [invalid])

    def test_new_crossing_and_already_enhanced_are_distinct(self) -> None:
        issues = [make_issue("synthetic", BASE), make_issue("synthetic", BASE + timedelta(hours=3))]
        observations = [
            Observation(BASE - timedelta(hours=1), 1.0),
            Observation(BASE + timedelta(hours=2), 12.0),
            Observation(BASE + timedelta(hours=3), 15.0),
        ]
        targets = build_targets(issues, observations)
        self.assertEqual(targets[0].label, 1)
        self.assertEqual(targets[1].excluded_reason, "already_above_threshold")

    def test_below_threshold_sample_separates_physical_episodes(self) -> None:
        issue_a = make_issue("synthetic", BASE)
        issue_b = make_issue("synthetic", BASE + timedelta(hours=4))
        observations = [
            Observation(BASE - timedelta(hours=1), 1.0),
            Observation(BASE + timedelta(hours=1), 12.0),
            Observation(BASE + timedelta(hours=2), 1.0),
            Observation(BASE + timedelta(hours=3), 1.0),
            Observation(BASE + timedelta(hours=5), 13.0),
        ]
        targets = build_targets([issue_a, issue_b], observations)
        units = build_cohort_units([issue_a, issue_b], targets, observations)
        episodes = [unit for unit in units if unit.kind == "episode"]
        self.assertEqual(len(episodes), 2)

    def test_partition_keeps_units_whole_and_enforces_purge(self) -> None:
        units = [
            CohortUnit(
                f"u-{index}",
                "episode" if index % 2 else "quiet_block",
                (f"i-{index}",),
                BASE + timedelta(hours=index * 72),
                BASE + timedelta(hours=index * 72 + 2),
                index % 2,
            )
            for index in range(5)
        ]
        roles = assign_chronological_roles(
            units,
            {role: 1 for role in ("train", "validation_monitor", "validation_calibration", "validation_threshold", "locked_test")},
        )
        self.assertEqual([roles[role][0].unit_id for role in roles if role != "purged"], [f"u-{i}" for i in range(5)])
        self.assertFalse(roles["purged"])

    def test_partition_drops_later_boundary_unit_inside_purge(self) -> None:
        units = [
            CohortUnit("train-unit", "quiet_block", ("a",), BASE, BASE + timedelta(hours=2), 0),
            CohortUnit("too-close", "episode", ("b",), BASE + timedelta(hours=12), BASE + timedelta(hours=13), 1),
        ]
        roles = assign_chronological_roles(
            units,
            {"train": 1, "validation_monitor": 1},
            purge_hours=24,
        )
        self.assertEqual(tuple(unit.unit_id for unit in roles["purged"]), ("too-close",))
        self.assertFalse(roles["validation_monitor"])

    def test_partition_drops_unit_at_inclusive_horizon_endpoint(self) -> None:
        units = [
            CohortUnit("train-unit", "quiet_block", ("a",), BASE, BASE, 0),
            CohortUnit("at-endpoint", "episode", ("b",), BASE + timedelta(hours=24), BASE + timedelta(hours=25), 1),
        ]
        roles = assign_chronological_roles(
            units,
            {"train": 1, "validation_monitor": 1},
            purge_hours=24,
        )
        self.assertEqual(tuple(unit.unit_id for unit in roles["purged"]), ("at-endpoint",))
        self.assertFalse(roles["validation_monitor"])

    def test_transform_fit_is_train_only_and_retains_missing_mask(self) -> None:
        standardizer = TrainOnlyStandardizer()
        train = [
            FeatureRow("a", "train", {"x": 1.0, "y": None}),
            FeatureRow("b", "train", {"x": 3.0, "y": 4.0}),
        ]
        receipt = standardizer.fit(train)
        self.assertEqual(receipt.fit_role, "train")
        transformed = standardizer.transform([FeatureRow("c", "validation_monitor", {"x": 2.0, "y": None})])
        self.assertEqual(transformed[0].observed_mask, (True, False))
        with self.assertRaises(ProtectedDataError):
            TrainOnlyStandardizer().fit([FeatureRow("z", "validation_monitor", {"x": 1.0})])

    def test_manifest_hash_detects_mutation_and_write_is_immutable(self) -> None:
        manifest = freeze_manifest("synthetic_cohort", {"issue_ids": ["a", "b"]})
        self.assertTrue(verify_manifest(manifest))
        changed = dict(manifest)
        changed["payload"] = {"issue_ids": ["a"]}
        self.assertFalse(verify_manifest(changed))
        with tempfile.TemporaryDirectory(prefix="iris_sep_manifest_") as directory:
            path = Path(directory) / "manifest.json"
            write_immutable_manifest(path, manifest)
            write_immutable_manifest(path, manifest)
            with self.assertRaisesRegex(PipelineError, "different content"):
                write_immutable_manifest(path, freeze_manifest("synthetic_cohort", {"issue_ids": ["x"]}))


if __name__ == "__main__":
    unittest.main()
