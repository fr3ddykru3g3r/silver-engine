"""Static tests that do not import PyTorch."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


MODEL = Path(__file__).resolve().parents[1] / "src" / "iris_sep" / "modeling" / "tabular_multibranch.py"


class TabularModelStaticTests(unittest.TestCase):
    def test_primary_model_is_schema_aligned_and_has_no_secondary_heads(self) -> None:
        text = MODEL.read_text(encoding="utf-8")
        ast.parse(text)
        for token in ("magnetic", "eruption", "particle_context", "observed_mask", "primary_head", "all_missing"):
            self.assertIn(token, text)
        for forbidden in ("peak_flux_head", "onset_hazard_head", "aia", "CausalConv1d", "locked_test"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
