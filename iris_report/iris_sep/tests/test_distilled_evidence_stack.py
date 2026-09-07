from __future__ import annotations

import unittest

import numpy as np

from iris_report.iris_sep.src.iris_sep.modeling.distilled_evidence_stack import (
    DistilledEvidenceStackConfig,
    DistilledPositiveEvidenceStack,
)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


class DistilledEvidenceStackTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([
            [-1.0, -0.8],
            [-0.7, -0.2],
            [-0.3,  0.1],
            [ 0.2,  0.4],
            [ 0.7,  0.9],
            [ 1.0,  1.3],
        ], dtype=float)
        self.y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        self.teacher = sigmoid(-0.1 + 0.9 * self.x[:, 0] + 0.7 * self.x[:, 1])

    def test_fit_has_nonnegative_weights_and_finite_scores(self):
        model = DistilledPositiveEvidenceStack(
            DistilledEvidenceStackConfig(expert_count=2, teacher_weight=0.35, l2_weight=0.03)
        ).fit(self.x, self.y, self.teacher)
        self.assertTrue(all(weight >= 0 for weight in model.fit_.weights))
        z = model.decision_function(self.x)
        self.assertTrue(np.isfinite(z).all())
        diagnostics = model.diagnostics()
        self.assertAlmostEqual(diagnostics["teacher_weight"], 0.35)
        self.assertAlmostEqual(diagnostics["hard_label_weight"], 0.65)

    def test_teacher_target_must_be_probability(self):
        model = DistilledPositiveEvidenceStack(DistilledEvidenceStackConfig(expert_count=2))
        bad = self.teacher.copy()
        bad[0] = 1.2
        with self.assertRaises(ValueError):
            model.fit(self.x, self.y, bad)

    def test_runtime_evidence_dimension_is_enforced(self):
        model = DistilledPositiveEvidenceStack(DistilledEvidenceStackConfig(expert_count=2)).fit(
            self.x, self.y, self.teacher
        )
        with self.assertRaises(ValueError):
            model.decision_function(np.column_stack([self.x, np.ones(len(self.x))]))

    def test_teacher_weight_cannot_replace_hard_labels_entirely(self):
        with self.assertRaises(ValueError):
            DistilledEvidenceStackConfig(expert_count=2, teacher_weight=1.0)


if __name__ == "__main__":
    unittest.main()
