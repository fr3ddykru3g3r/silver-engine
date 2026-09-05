from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - dependency guard for source-only environments
    raise unittest.SkipTest("PyTorch is required for the inner neural tests")
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iris_report.iris_sep.tools.train_tabular_multibranch import TARGET
from iris_report.iris_sep.workstreams.luna_inner_neural_20260905.helper import fit_predict


def _frame(n=12):
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        "role": ["train"] * n, TARGET: np.arange(n) % 2,
        "sharp_mean": rng.normal(size=n), "flare_count": rng.normal(size=n),
        "protonflux_mean": rng.normal(size=n),
    })


class TestInnerNeural(unittest.TestCase):
    def test_rejects_any_non_train_role(self):
        frame = _frame()
        frame.loc[0, "role"] = "validation_monitor"
        with self.assertRaisesRegex(ValueError, "only role=train"):
            fit_predict(frame, ["sharp_mean", "flare_count"], [1, 2, 3, 4], [5, 6, 7, 8], [9], 7, self._out("role"), max_epochs=1)


    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _out(self, name):
        return Path(self.tmp.name) / name

    def test_preprocessing_is_fit_only_on_fit_indices(self):
        frame = _frame()
        features = ["sharp_mean", "flare_count"]
        fit = np.array([0, 1, 2, 3, 4, 5])
        stop = np.array([6, 7, 8, 9])
        pred = np.array([10, 11])
        first_dir = self._out("first")
        fit_predict(frame, features, fit, stop, pred, 7, first_dir, max_epochs=1)
        changed = frame.copy()
        changed.loc[stop, features] = 1_000_000
        second_dir = self._out("second")
        fit_predict(changed, features, fit, stop, pred, 7, second_dir, max_epochs=1)
        self.assertEqual((first_dir / "preprocessing.json").read_bytes(), (second_dir / "preprocessing.json").read_bytes())


    def test_fit_and_stop_must_be_disjoint(self):
        frame = _frame()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_predict(frame, ["sharp_mean", "flare_count"], [0, 1, 2, 3, 4, 5], [5, 6, 7, 8], [9], 7, self._out("overlap"), max_epochs=1)
