from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from iris_report.iris_sep.src.iris_sep.promoted_model_package import (
    PromotedModelPackageError,
    export_promoted_package,
    load_promoted_package,
)


class PromotedModelPackageTests(unittest.TestCase):
    def _fixture(self, root: Path):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame({
            "s1": rng.normal(size=60),
            "x1": rng.normal(size=60),
            "p1": rng.normal(size=60),
        })
        y = ((frame.s1 + 0.3 * frame.x1 - 0.1 * frame.p1) > 0).astype(int).to_numpy()
        family_models = {}
        for family, feature in (("SOLAR", "s1"), ("XRS", "x1"), ("PROTON", "p1")):
            model = XGBClassifier(
                n_estimators=4,
                max_depth=2,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=1,
                random_state=3,
            )
            model.fit(frame[[feature]], y, verbose=False)
            family_models[family] = [model] * 5
        out = root / "package"
        export_promoted_package(
            output_dir=out,
            family_models=family_models,
            feature_families={"SOLAR": ["s1"], "XRS": ["x1"], "PROTON": ["p1"]},
            fit_prevalence=float(y.mean()),
            stack_intercept=-0.2,
            stack_weights=[0.2, 0.3, 0.4],
            evidence_limit=6.0,
            calibration_intercept=0.1,
            thresholds={"MAX_TSS": 0.2, "POD80_MIN_FAR": 0.1},
            source_bindings={"fixture": "abc"},
            dependency_versions={"fixture": "1"},
            training_receipt={"locked_test_accessed": False},
        )
        return frame, out

    def test_round_trip_is_load_only_and_finite(self):
        with tempfile.TemporaryDirectory() as temp:
            frame, package = self._fixture(Path(temp))
            loaded = load_promoted_package(package)
            result = loaded.predict(frame)
            self.assertEqual(result["probability"].shape, (len(frame),))
            self.assertTrue(np.isfinite(result["probability"]).all())
            self.assertTrue(((result["probability"] >= 0) & (result["probability"] <= 1)).all())
            self.assertEqual(set(loaded.thresholds), {"MAX_TSS", "POD80_MIN_FAR"})
            self.assertFalse(loaded.manifest["runtime_training_allowed"])

    def test_model_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, package = self._fixture(Path(temp))
            model_path = package / "models" / "solar" / "seed_0.json"
            model_path.write_bytes(model_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(PromotedModelPackageError, "model file digest mismatch"):
                load_promoted_package(package)

    def test_manifest_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            _, package = self._fixture(Path(temp))
            manifest = json.loads((package / "manifest.json").read_text())
            manifest["thresholds"]["MAX_TSS"] = 0.999
            (package / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
            with self.assertRaisesRegex(PromotedModelPackageError, "manifest digest mismatch"):
                load_promoted_package(package)

    def test_missing_feature_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            frame, package = self._fixture(Path(temp))
            loaded = load_promoted_package(package)
            with self.assertRaisesRegex(PromotedModelPackageError, "required package features missing"):
                loaded.predict(frame.drop(columns=["x1"]))

    def test_export_requires_exactly_five_models_per_family(self):
        with tempfile.TemporaryDirectory() as temp:
            rng = np.random.default_rng(4)
            frame = pd.DataFrame({"s1": rng.normal(size=20)})
            y = (frame.s1 > 0).astype(int).to_numpy()
            model = XGBClassifier(n_estimators=2, n_jobs=1, random_state=1)
            model.fit(frame[["s1"]], y, verbose=False)
            with self.assertRaisesRegex(PromotedModelPackageError, "exactly 5"):
                export_promoted_package(
                    output_dir=Path(temp) / "bad",
                    family_models={"SOLAR": [model], "XRS": [model] * 5, "PROTON": [model] * 5},
                    feature_families={"SOLAR": ["s1"], "XRS": ["x1"], "PROTON": ["p1"]},
                    fit_prevalence=0.5,
                    stack_intercept=0.0,
                    stack_weights=[0.1, 0.1, 0.1],
                    evidence_limit=6.0,
                    calibration_intercept=0.0,
                    thresholds={"MAX_TSS": 0.5, "POD80_MIN_FAR": 0.4},
                    source_bindings={},
                    dependency_versions={},
                    training_receipt={},
                )


if __name__ == "__main__":
    unittest.main()
