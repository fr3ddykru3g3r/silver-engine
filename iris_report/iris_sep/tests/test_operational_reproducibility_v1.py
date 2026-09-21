import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_operational_reproducibility_v1.py"
spec = importlib.util.spec_from_file_location("operational_audit", TOOL)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _families(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps({"families": [
        {"family": "SHARP_AR", "prefixes": ["SHARP_AR_"], "status": "SCHEMA_MISMATCH"},
        {"family": "SHARP", "prefixes": ["SHARP_"], "status": "SCHEMA_MISMATCH"},
        {"family": "PROTON", "prefixes": ["ProtonFlux_"], "status": "VERIFIED"},
        {"family": "XRS", "prefixes": ["XRSB_"], "status": "UNVERIFIED_LATENCY"}
    ]}), encoding="utf-8")
    return mod.load_manifest(p)


def test_longest_prefix_wins(tmp_path):
    families = _families(tmp_path)
    assert mod.classify("SHARP_AR_USFLUX_max", families)["family"] == "SHARP_AR"
    assert mod.classify("SHARP_USFLUX_max", families)["family"] == "SHARP"


def test_nonpredictor_columns_are_excluded(tmp_path):
    families = _families(tmp_path)
    result = mod.audit([
        "window_begin", "window_end", "Future_OSEP_label", "OSEP_label",
        "ProtonFlux_max", "XRSB_avg"
    ], families)
    assert result["predictor_count"] == 2
    assert result["verified_predictor_count"] == 1
    assert result["operationally_reproducible_feature_fraction"] == 0.5


def test_unknown_predictor_fails_closed(tmp_path):
    families = _families(tmp_path)
    with pytest.raises(ValueError, match="Unmapped predictor"):
        mod.audit(["UnknownSensor_value"], families)


def test_status_counts_are_explicit(tmp_path):
    families = _families(tmp_path)
    result = mod.audit([
        "ProtonFlux_max", "XRSB_avg", "SHARP_USFLUX_max", "SHARP_AR_USFLUX_max"
    ], families)
    assert result["status_counts"] == {
        "SCHEMA_MISMATCH": 2,
        "UNVERIFIED_LATENCY": 1,
        "VERIFIED": 1,
    }
