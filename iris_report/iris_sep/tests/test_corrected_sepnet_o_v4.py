from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from iris_report.iris_sep.tools import train_corrected_sepnet_o_v4 as adapter


class CorrectedSEPNetOV3Tests(unittest.TestCase):
    def test_architecture_and_single_shared_forward(self) -> None:
        model = adapter.SEPNetDense()
        linear = [module for module in model.shared if isinstance(module, torch.nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in linear], [(98, 256), (256, 128), (128, 64), (64, 16)])
        self.assertEqual(model.reg_head.in_features, 16)
        self.assertEqual(model.cls_head.in_features, 16)
        reg, cls = model(torch.zeros(3, 98))
        self.assertEqual(tuple(reg.shape), (3,)); self.assertEqual(tuple(cls.shape), (3,))
        self.assertEqual(inspect.getsource(adapter.SEPNetDense.forward).count("self.shared("), 1)

    def test_backprop_precedes_gradient_clipping(self) -> None:
        source = inspect.getsource(adapter.run)
        self.assertLess(source.index("loss.backward()"), source.index("clip_grad_norm_"))
        self.assertLess(source.index("clip_grad_norm_"), source.index("optimizer.step()"))

    def test_v6_only_and_role_contract_is_static(self) -> None:
        self.assertEqual(adapter.sha256_file(adapter.SOURCE), adapter.SOURCE_SHA256)
        self.assertEqual(adapter.sha256_file(adapter.SOURCE_MANIFEST), adapter.MANIFEST_SHA256)
        frame = pd.read_csv(adapter.SOURCE, nrows=1)
        features = [column for column in frame.columns if column not in adapter.META]
        self.assertEqual(len(features), 98)
        self.assertNotIn(adapter.OPERATIONAL, features)
        source = inspect.getsource(adapter.run)
        self.assertIn('fit_intercept_calibration', source)
        self.assertIn('role="validation_calibration"', source)
        self.assertIn('role="validation_threshold"', source)
        self.assertNotIn("locked_test", set(pd.read_csv(adapter.SOURCE, usecols=["role"])["role"]))

    def test_loss_has_both_heads_and_finite_gradients(self) -> None:
        model = adapter.SEPNetDense(); x = torch.randn(8, 98); ycls = torch.tensor([0, 1] * 4, dtype=torch.float32); yreg = torch.arange(8, dtype=torch.float32); weights = torch.ones(8)
        reg, logits = model(x); loss = adapter._loss(reg, logits, yreg, ycls, weights); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_episode_balance_equalizes_unit_contribution(self) -> None:
        frame = pd.DataFrame({"role":["train"]*4,"window_begin":pd.date_range("2020-01-01",periods=4),"future_SEP_label":[1,1,1,0],"unit_id":["a","a","a","b"],"future_Operational_SEP_label":[0,0,0,1]})
        weights, mapping = adapter.episode_weights(frame, "episode_balanced")
        self.assertEqual(weights.shape, (4,)); self.assertTrue(np.isfinite(weights).all()); self.assertNotIn('frame["unit_id"]', inspect.getsource(adapter.general_episode_mapping))
        mutated = frame.copy(); mutated["future_Operational_SEP_label"] = 1 - mutated["future_Operational_SEP_label"]
        self.assertTrue(np.array_equal(weights, adapter.episode_weights(mutated, "episode_balanced")[0]))

    def test_adversarial_validation_features_do_not_change_train_preprocessing(self) -> None:
        frame = pd.read_csv(adapter.SOURCE)
        features = [c for c in frame.columns if c not in adapter.META]
        altered = frame.copy(); validation = altered.role != "train"; altered.loc[validation, features] = 999999.0; altered.loc[validation, adapter.GENERAL] = 1 - altered.loc[validation, adapter.GENERAL]
        original = adapter.fit_preprocessing(frame, features)
        mutated = adapter.fit_preprocessing(altered, features)
        self.assertEqual(original[2], mutated[2])
        self.assertEqual(adapter.preprocessing_train_fingerprint(frame, features), adapter.preprocessing_train_fingerprint(altered, features))

    def test_integrity_contract_is_present(self) -> None:
        source = inspect.getsource(adapter.run)
        for token in ("run_config_sha256", "observed_feature_mask_sha256", "training_weights_sha256", "general_episode_mapping_sha256", "round_trip_reloaded_development_predictions.csv", "INTERRUPTED_AFTER_CHECKPOINT"):
            self.assertIn(token, source)

    def test_interrupted_resume_is_bitwise_equivalent(self) -> None:
        with TemporaryDirectory() as temp:
            interrupted = Path(temp) / "interrupted"; complete = Path(temp) / "complete"
            with patch.object(adapter, "SEEDS", (7,)):
                with self.assertRaisesRegex(RuntimeError, "INTERRUPTED_AFTER_CHECKPOINT"):
                    adapter.run("faithful_row_weighted", interrupted, max_epochs=2, patience=20, interrupt_after=1)
                checkpoint = torch.load(interrupted / "seed_7" / "last.pt", map_location="cpu", weights_only=False)
                for key in ("optimizer", "scheduler", "stale", "rng", "best_loss", "best_epoch"):
                    self.assertIn(key, checkpoint)
                adapter.run("faithful_row_weighted", interrupted, max_epochs=2, patience=20, resume=True)
                adapter.run("faithful_row_weighted", complete, max_epochs=2, patience=20)
            self.assertEqual((interrupted / "development_predictions.csv").read_bytes(), (complete / "development_predictions.csv").read_bytes())

    def test_claims_and_modes_are_separate_in_smoke_receipt_contract(self) -> None:
        self.assertNotEqual("faithful_row_weighted", "episode_balanced")
        source = inspect.getsource(adapter.run)
        self.assertIn('FAITHFUL_CORRECTED_SEPNET_O', source)
        self.assertIn('PREDECLARED_IRIS_EPISODE_BALANCED_EXPERIMENT', source)
        for claim in ("SUPERIORITY", "PRODUCTION_READINESS", "SEPVAL_SCORE"):
            self.assertIn(claim, source)


if __name__ == "__main__":
    unittest.main()
