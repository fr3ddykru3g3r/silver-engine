"""Run the IRIS-SEP checks that do not require a GPU or locked data."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


IRIS_SEP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = IRIS_SEP_ROOT.parents[1]


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


run([sys.executable, "tests/verify_contract.py"], cwd=IRIS_SEP_ROOT)
run(
    [sys.executable, "-m", "unittest", "test_baseline_interface.py", "-v"],
    cwd=IRIS_SEP_ROOT / "workstreams" / "luna_b",
)
run(
    [sys.executable, "-m", "unittest", "test_flare_system.py", "-v"],
    cwd=WORKSPACE / "iris_report" / "source_snapshot" / "iris-model",
)
run(
    [
        sys.executable,
        "-m",
        "unittest",
        "iris_report.iris_sep.workstreams.luna_d.test_identity_bridge",
        "iris_report.iris_sep.workstreams.luna_d.test_aia_pretraining",
        "-v",
    ],
    cwd=WORKSPACE,
)
run(
    [
        sys.executable,
        "-m",
        "unittest",
        "iris_report.iris_sep.tests.test_prepare_sepnet_v1_dual_target_v6",
        "iris_report.iris_sep.tests.test_corrected_sepnet_o_v1",
        "iris_report.iris_sep.tests.test_corrected_sepnet_o_v5",
        "iris_report.iris_sep.tests.test_seal_training_split",
        "iris_report.iris_sep.tests.test_pilot_replay",
        "iris_report.iris_sep.tests.test_inner_training_diagnostic",
        "iris_report.iris_sep.tests.test_validity_envelope_benchmark",
        "iris_report.iris_sep.tests.test_pilot_admission_v2",
        "iris_report.iris_sep.tests.test_compound_validity_benchmark",
        "iris_report.iris_sep.workstreams.luna_inner_neural_20260905.test_helper",
        "iris_report.iris_sep.tests.test_tabular_model_static",
        "iris_report.iris_sep.tests.test_tabular_model_runtime",
        "-v",
    ],
    cwd=WORKSPACE,
)
run(
    [sys.executable, "workstreams/luna_c/verify_static.py"],
    cwd=IRIS_SEP_ROOT,
)
run(
    [
        sys.executable,
        "-m",
        "unittest",
        "iris_report.iris_sep.workstreams.luna_g_data_pipeline.test_pipeline_unittest",
        "iris_report.iris_sep.workstreams.luna_h_model_hardening.test_contract_unittest",
        "iris_report.iris_sep.workstreams.luna_i_eval_ops.test_eval_ops_unittest",
        "-v",
    ],
    cwd=WORKSPACE,
)

for path in (IRIS_SEP_ROOT / "workstreams" / "luna_a").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for workstream in (
    "luna_c",
    "luna_d",
    "luna_g_data_pipeline",
    "luna_h_model_hardening",
    "luna_i_eval_ops",
):
    for path in (IRIS_SEP_ROOT / "workstreams" / workstream).rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for source_root in (IRIS_SEP_ROOT / "src", IRIS_SEP_ROOT / "tools", IRIS_SEP_ROOT / "colab"):
    for path in source_root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

checkpoint = json.loads(
    (IRIS_SEP_ROOT / "receipts" / "colab_live_checkpoint_2026-09-04.json").read_text()
)
assert checkpoint["drive_file"]["notebook_sha256"] == sha256(
    WORKSPACE / "iris_report" / "colab" / "IRIS_Quantum_Flare_Benchmark_2026-09-02.ipynb"
)
assert checkpoint["builder_sha256"] == sha256(
    WORKSPACE / "iris_report" / "colab" / "build_quantum_flare_notebook.py"
)
assert checkpoint["locked_sep_test_accessed"] is False

selected_split = json.loads(
    (IRIS_SEP_ROOT / "receipts" / "development_split_selection_2026-09-04.json").read_text()
)
selected_manifest = IRIS_SEP_ROOT / "receipts" / selected_split["selected_candidate"]["manifest"]
assert selected_split["locked_test_accessed"] is False
assert selected_split["status"] == "DEVELOPMENT_ONLY_NOT_FINAL_BENCHMARK"
assert selected_split["selected_candidate"]["manifest_sha256"] == sha256(selected_manifest)
for rejected_candidate in selected_split["rejected_candidates"]:
    rejected_manifest = IRIS_SEP_ROOT / "receipts" / rejected_candidate["manifest"]
    assert rejected_candidate["manifest_sha256"] == sha256(rejected_manifest)

# Authoritative dual-target development adapter.  These literals make an
# unreviewed artifact or source mutation fail the top-level verifier.
dual_v6_csv = IRIS_SEP_ROOT / "data_processed" / "sepnet_v1_development_v6_dual_target.csv"
dual_v6_manifest_path = IRIS_SEP_ROOT / "receipts" / "sepnet_v1_development_v6_dual_target_manifest.json"
assert sha256(dual_v6_manifest_path) == "72f92f18936e237e9817fc4b425f2ebfec0096356b7a91efd65fbcf6852f4052"
dual_v6 = json.loads(dual_v6_manifest_path.read_text())
assert dual_v6["status"] == "DEVELOPMENT_ONLY_LEGACY_DUAL_TARGET_NOT_FINAL_BENCHMARK"
assert dual_v6["output_sha256"] == "cc5ea62ff0a8423b8b9e3c028487dd61704fdeb950e84d4968b278179b841d04"
assert dual_v6["output_sha256"] == sha256(dual_v6_csv)
assert dual_v6["publisher_training_source_sha256"] == sha256(
    IRIS_SEP_ROOT / dual_v6["publisher_training_source_repo_relative"]
)
assert dual_v6["frozen_v3_csv_sha256"] == sha256(IRIS_SEP_ROOT / dual_v6["frozen_v3_csv_repo_relative"])
assert dual_v6["frozen_v3_manifest_sha256"] == sha256(
    IRIS_SEP_ROOT / dual_v6["frozen_v3_manifest_repo_relative"]
)
json_hash = lambda value: hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
assert dual_v6["ordered_feature_schema_sha256"] == "7bca82f223f1be0adbd8afc6e30aed238ed52b3bb2339a98fa9c9cbd944436b5"
assert dual_v6["ordered_feature_schema_sha256"] == json_hash(dual_v6["ordered_feature_columns"])
assert dual_v6["ordered_target_schema_sha256"] == "93483afb5505aab2af024052b597cb2d953f4114060d7763945439dd3c61eeec"
assert dual_v6["ordered_target_schema_sha256"] == json_hash(tuple(dual_v6["ordered_target_schema"]))
dual_frame = pd.read_csv(dual_v6_csv, float_precision="round_trip")
mapping_bytes = dual_frame.loc[:, dual_v6["frozen_v3_mapping_columns"]].to_csv(
    index=False, lineterminator="\n"
).encode()
assert hashlib.sha256(mapping_bytes).hexdigest() == dual_v6["frozen_v3_mapping_sha256"]
source_paths = {
    "v6_tool": IRIS_SEP_ROOT / "tools" / "prepare_sepnet_v1_dual_target_development_v6.py",
    "v5_tool": IRIS_SEP_ROOT / "tools" / "prepare_sepnet_v1_dual_target_development_v5.py",
    "v4_dual_target_builder": IRIS_SEP_ROOT / "tools" / "prepare_sepnet_v1_dual_target_development.py",
    "v3_prepare_and_build_units": IRIS_SEP_ROOT / "tools" / "prepare_sepnet_v1_development.py",
    "cohort_assignment_implementation": IRIS_SEP_ROOT / "workstreams" / "luna_g_data_pipeline" / "iris_sep_pipeline" / "cohort.py",
}
assert dual_v6["source_code_sha256"] == {name: sha256(path) for name, path in source_paths.items()}
assert dual_v6["source_code_sha256"]["v6_tool"] == "b2bc98dc9c0262e19129f31145f49c739952b732578011a6bc24eba4655b6ac9"
assert dual_v6["testing_or_sepval_artifact_accessed"] is False
assert {"superiority", "production_readiness"}.issubset(dual_v6["forbidden_claims"])

for directory, expected_receipt, expected_mode, expected_role in (
    ("corrected_sepnet_o_v1_faithful_v2", "a21bfd4135343f7909487bddb23c71d6ba0a37a3423576f26b272d537c35d672", "faithful_row_weighted", "FAITHFUL_CORRECTED_SEPNET_O"),
    ("corrected_sepnet_o_v1_episode_balanced_v2", "0d4f1c545c232ee048f6182996672adc01cf78038407eed853e9d90066a47248", "episode_balanced", "PREDECLARED_IRIS_EPISODE_BALANCED_EXPERIMENT"),
):
    sepnet_dir = IRIS_SEP_ROOT / "artifacts" / directory
    sepnet_receipt_path = sepnet_dir / "receipt.json"
    assert sha256(sepnet_receipt_path) == expected_receipt
    sepnet_receipt = json.loads(sepnet_receipt_path.read_text())
    assert sepnet_receipt["experiment_mode"] == expected_mode
    assert sepnet_receipt["comparator_role"] == expected_role
    assert sepnet_receipt["architecture"] == [98, 256, 128, 64, 16]
    assert sepnet_receipt["one_shared_forward_pass"] is True
    assert sepnet_receipt["training_labels"]["operational_label_used_for_training"] is False
    assert sepnet_receipt["role_contract"]["calibration"] == "validation_calibration_operational_label"
    assert sepnet_receipt["role_contract"]["threshold"] == "validation_threshold_operational_label"
    assert sepnet_receipt["source_sha256"] == dual_v6["output_sha256"]
    assert sepnet_receipt["source_manifest_sha256"] == sha256(dual_v6_manifest_path)
    assert sepnet_receipt["source_code_sha256"]["adapter"] == sha256(
        IRIS_SEP_ROOT / "tools" / "train_corrected_sepnet_o_v1.py"
    )
    assert sepnet_receipt["predictions_sha256"] == sha256(sepnet_dir / "development_predictions.csv")
    assert sepnet_receipt["preprocessing_sha256"] == sha256(sepnet_dir / "preprocessing.pkl")
    assert sepnet_receipt["preprocessing_receipt_sha256"] == sha256(sepnet_dir / "preprocessing_receipt.json")
    assert (sepnet_dir / "observed_feature_mask.npz").is_file()
    assert (sepnet_dir / "run_config.json").is_file()
    assert sepnet_receipt["locked_test_accessed"] is False
    assert sepnet_receipt["headline_eligible_roles"] == []
    assert {"SUPERIORITY", "PRODUCTION_READINESS"}.issubset(sepnet_receipt["claims_forbidden"])
    for seed_receipt in sepnet_receipt["seeds"]:
        assert seed_receipt["best_checkpoint_sha256"] == sha256(sepnet_dir / seed_receipt["best_checkpoint"])
        assert seed_receipt["last_checkpoint_sha256"] == sha256(sepnet_dir / seed_receipt["last_checkpoint"])

baseline_selection = json.loads(
    (IRIS_SEP_ROOT / "receipts" / "local_baseline_selection_2026-09-04.json").read_text()
)
baseline_receipt_path = IRIS_SEP_ROOT / baseline_selection["authoritative_development_artifact"]["path"]
assert baseline_selection["authoritative_development_artifact"]["receipt_sha256"] == sha256(baseline_receipt_path)
baseline_receipt = json.loads(baseline_receipt_path.read_text())
assert baseline_receipt["status"] == "PASS_DEVELOPMENT_ONLY"
assert baseline_receipt["locked_test_accessed"] is False
assert baseline_receipt["source_sha256"] == selected_split["selected_candidate"]["output_sha256"]
assert baseline_receipt["seeds"] == [7, 13, 26, 42, 73]
assert set(baseline_receipt["claims_forbidden"]) == {
    "SEPVAL_SCORE", "FINAL_NEW_CROSSING_SCORE", "BREAKTHROUGH", "OPERATIONAL_CERTIFICATION"
}
assert baseline_receipt["predictions_sha256"] == sha256(
    baseline_receipt_path.parent / "development_predictions.csv"
)
for baseline_run in baseline_receipt["runs"]:
    assert set(baseline_run["metrics"]) == {
        "train", "validation_monitor", "validation_calibration", "validation_threshold"
    }
    if baseline_run["model_artifact"] is not None:
        assert baseline_run["model_artifact_sha256"] == sha256(
            baseline_receipt_path.parent / baseline_run["model_artifact"]
        )

for artifact_key in ("authoritative_xgboost_artifact", "authoritative_ensemble_artifact"):
    artifact = baseline_selection[artifact_key]
    artifact_path = IRIS_SEP_ROOT / artifact["path"]
    assert artifact["receipt_sha256"] == sha256(artifact_path)
    artifact_receipt = json.loads(artifact_path.read_text())
    assert artifact_receipt["locked_test_accessed"] is False
    assert artifact_receipt["headline_eligible_roles"] == []

neural_selection_path = IRIS_SEP_ROOT / "receipts" / "local_neural_selection_2026-09-05.json"
neural_selection = json.loads(neural_selection_path.read_text())
assert neural_selection["status"] == "DEVELOPMENT_ONLY_NOT_HEADLINE_ELIGIBLE"
assert neural_selection["locked_test_accessed"] is False
neural_artifact = neural_selection["authoritative_neural_artifact"]
neural_receipt_path = IRIS_SEP_ROOT / neural_artifact["path"]
assert neural_artifact["receipt_sha256"] == sha256(neural_receipt_path)
neural_receipt = json.loads(neural_receipt_path.read_text())
assert neural_receipt["status"] == "PASS_DEVELOPMENT_ONLY"
assert neural_receipt["locked_test_accessed"] is False
assert neural_receipt["headline_eligible_roles"] == []
assert neural_receipt["source_sha256"] == selected_split["selected_candidate"]["output_sha256"]
assert neural_receipt["predictions_sha256"] == sha256(
    neural_receipt_path.parent / "development_predictions.csv"
)
assert neural_receipt["preprocessing_sha256"] == sha256(
    neural_receipt_path.parent / "preprocessing.json"
)
for seed_run in neural_receipt["seeds"]:
    assert seed_run["best_checkpoint_sha256"] == sha256(
        neural_receipt_path.parent / seed_run["best_checkpoint"]
    )
comparison = neural_selection["authoritative_comparison"]
comparison_path = IRIS_SEP_ROOT / comparison["path"]
assert comparison["receipt_sha256"] == sha256(comparison_path)
comparison_receipt = json.loads(comparison_path.read_text())
assert comparison_receipt["locked_test_accessed"] is False
assert comparison_receipt["paired_ci_lower_above_zero"] is False
assert comparison_receipt["status"] == "DEVELOPMENT_DIAGNOSTIC_INCONCLUSIVE"
hybrid = neural_selection["fixed_hybrid_development_candidate"]
hybrid_path = IRIS_SEP_ROOT / hybrid["path"]
assert hybrid["receipt_sha256"] == sha256(hybrid_path)
hybrid_receipt = json.loads(hybrid_path.read_text())
assert hybrid_receipt["status"] == "DEVELOPMENT_DIAGNOSTIC_INCONCLUSIVE"
assert hybrid_receipt["locked_test_accessed"] is False
assert hybrid_receipt["superiority_claim"] is False
assert hybrid_receipt["frozen_candidate"]["iris_probability_weight"] == 0.5
assert hybrid_receipt["frozen_candidate"]["xgboost_probability_weight"] == 0.5
assert hybrid_receipt["frozen_candidate"]["threshold_fit_role"] == "validation_threshold"
assert hybrid_receipt["paired_unit_bootstrap_tss_difference"]["valid_replicates"] == 10_000
assert hybrid_receipt["paired_unit_bootstrap_tss_difference"]["ci_lower_95"] <= 0
assert hybrid_receipt["hashes"]["source_script_sha256"] == sha256(
    IRIS_SEP_ROOT / "tools" / "build_development_hybrid.py"
)
assert hybrid_receipt["hashes"]["predictions_sha256"] == sha256(
    hybrid_path.parent / "development_hybrid_predictions.csv"
)
assert hybrid_receipt["hashes"]["exploratory_grid_sha256"] == sha256(
    hybrid_path.parent / "exploratory_grid_selection_biased.csv"
)
for superseded in neural_selection["superseded_artifacts"]:
    assert superseded["receipt_sha256"] == sha256(IRIS_SEP_ROOT / superseded["path"])

for path in IRIS_SEP_ROOT.rglob("*.json"):
    json.loads(path.read_text(encoding="utf-8"))

sepnet_status = json.loads(
    (IRIS_SEP_ROOT / "receipts" / "sepnet_reproduction_status_2026-09-05.json").read_text()
)
assert sepnet_status["status"] == "NOT_REPRODUCED"
assert sepnet_status["locked_test_accessed"] is False
assert "SEPNET_REPRODUCED" in sepnet_status["claims_forbidden"]
v2_access = json.loads(
    (IRIS_SEP_ROOT / "receipts" / "v2_clear_training_access_status_2026-09-05.json").read_text()
)
assert v2_access["status"] == "BLOCKED_PENDING_TRAINING_ONLY_ARTIFACT"
assert v2_access["locked_test_accessed"] is False
assert v2_access["large_archive_downloaded"] is False
assert "download the full table and inspect or locally filter locked rows" in v2_access["prohibited_workarounds"]

colab_package = json.loads(
    (IRIS_SEP_ROOT / "receipts" / "colab_development_package_2026-09-04.json").read_text()
)
colab_notebook = IRIS_SEP_ROOT / colab_package["notebook_path"]
colab_builder = IRIS_SEP_ROOT / colab_package["builder_path"]
assert colab_package["execution_status"] == "READY_NOT_RUN"
assert colab_package["locked_test_accessed"] is False
assert colab_package["notebook_sha256"] == sha256(colab_notebook)
assert colab_package["builder_sha256"] == sha256(colab_builder)
assert colab_package["drive_upload"]["downloaded_sha256"] == sha256(colab_notebook)
assert colab_package["drive_upload"]["size_bytes"] == colab_notebook.stat().st_size
assert colab_package["source_hashes"]["model"] == sha256(
    IRIS_SEP_ROOT / "src" / "iris_sep" / "modeling" / "tabular_multibranch.py"
)
assert colab_package["source_hashes"]["trainer"] == sha256(
    IRIS_SEP_ROOT / "tools" / "train_tabular_multibranch.py"
)
notebook_payload = json.loads(colab_notebook.read_text())
assert len(notebook_payload["cells"]) == colab_package["notebook_cells"]
for index, cell in enumerate(notebook_payload["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if source.startswith("%pip"):
            source = "\n".join(source.splitlines()[1:])
        ast.parse(source, filename=f"{colab_notebook}:cell-{index}")
notebook_text = colab_notebook.read_text()
assert "@gmail.com" not in notebook_text
assert "rolling_combinded_testing.csv" not in notebook_text
assert "SEPValidationChallengePhaseIII" not in notebook_text

run([sys.executable, "-m", "iris_report.iris_sep.tests.verify_corrected_sepnet_v5"], cwd=WORKSPACE)
print("IRIS_SEP_LOCAL_VERIFICATION_PASS")
if importlib.util.find_spec("torch") is None:
    print("PYTORCH_RUNTIME_TESTS_SKIPPED: local PyTorch is unavailable")
else:
    print("PYTORCH_CPU_FORWARD_BACKWARD_CHECKPOINT_TESTS_PASS")
print("COLAB_GPU_FULL_TRAINING_NOT_RUN")
print("LOCKED_TEST_NOT_ACCESSED")
