import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.modeling.positive_evidence_stack import PositiveEvidenceStack


class PositiveEvidenceStackTests(unittest.TestCase):
    def test_weights_are_nonnegative_and_fit_is_deterministic(self):
        x = np.array([
            [-2.0, -1.0, -1.5], [-1.5, -0.5, -1.0], [-1.0, -1.0, -0.8],
            [1.0, 0.8, 0.5], [1.5, 1.2, 1.0], [2.0, 1.5, 1.8],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        a = PositiveEvidenceStack().fit(x, y)
        b = PositiveEvidenceStack().fit(x, y)
        self.assertTrue(all(w >= 0 for w in a.fit_.weights))
        np.testing.assert_allclose(a.decision_function(x), b.decision_function(x), atol=1e-12)

    def test_increasing_positive_evidence_cannot_reduce_score(self):
        x = np.array([[-2, -2, -2], [-1, -1, -1], [1, 1, 1], [2, 2, 2]], dtype=float)
        y = np.array([0, 0, 1, 1])
        model = PositiveEvidenceStack().fit(x, y)
        low = model.decision_function(np.array([[0.0, 0.0, 0.0]]))[0]
        high = model.decision_function(np.array([[1.0, 1.0, 1.0]]))[0]
        self.assertGreaterEqual(high, low)

    def test_nonfinite_evidence_fails_closed(self):
        with self.assertRaises(ValueError):
            PositiveEvidenceStack().fit(np.array([[0, 1, np.nan], [1, 2, 3.0]]), np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
