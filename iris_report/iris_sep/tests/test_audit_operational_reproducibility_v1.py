import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_operational_reproducibility_v1.py"
spec = importlib.util.spec_from_file_location("op_audit", TOOL)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_exact_json_schema_keeps_present_time_labels(tmp_path):
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"columns": ["SHARP_label", "XRS_label"]}))
    cols, _ = mod.load_columns(schema)
    assert cols == ["SHARP_label", "XRS_label"]


def test_csv_rule_keeps_current_labels_but_drops_targets(tmp_path):
    table = tmp_path / "x.csv"
    table.write_text("window_begin,window_end,OSEP_label,GSEP_label,SHARP_label,XRS_label,Future_OSEP_label\n")
    cols, _ = mod.load_columns(table)
    assert cols == ["SHARP_label", "XRS_label"]


def test_longest_prefix_prevents_sharp_ar_from_being_swallowed():
    fams = [
        {"family": "SHARP_AR", "prefixes": ["SHARP_AR_"], "status": "SCHEMA_MISMATCH"},
        {"family": "SHARP", "prefixes": ["SHARP_"], "status": "SCHEMA_MISMATCH"},
    ]
    loaded = sorted(fams, key=lambda f: max(len(p) for p in f["prefixes"]), reverse=True)
    assert mod.classify("SHARP_AR_USFLUX_avg", loaded)["family"] == "SHARP_AR"


def test_unverified_is_not_counted_as_directly_non_equivalent():
    fams = [
        {"family": "A", "prefixes": ["A_"], "status": "VERIFIED"},
        {"family": "B", "prefixes": ["B_"], "status": "UNVERIFIED_LATENCY"},
        {"family": "C", "prefixes": ["C_"], "status": "RETROSPECTIVE_ONLY"},
    ]
    r = mod.audit(["A_x", "B_x", "C_x"], fams)
    assert r["verified_predictor_count"] == 1
    assert r["directly_non_equivalent_or_retrospective_count"] == 1
    assert r["unresolved_predictor_count"] == 1
