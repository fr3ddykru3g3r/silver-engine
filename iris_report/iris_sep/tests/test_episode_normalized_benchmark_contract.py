from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "config" / "episode_normalized_causal_benchmark_v1_preregistration_2026-09-09.json"
REGISTRY = ROOT / "config" / "inspected_evidence_registry_v2.json"
DESIGN = ROOT / "architecture" / "EPISODE_NORMALIZED_CAUSAL_BENCHMARK_2026-09-09.md"
CURRENT = ROOT / "CURRENT_STATUS.md"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_preregistration_freezes_episode_normalized_design_before_protected_outcomes() -> None:
    data = _load(PREREG)
    assert data["study_id"] == "IRIS_EPISODE_NORMALIZED_CAUSAL_BENCHMARK_V1"
    assert data["status"] == "PREREGISTERED_DESIGN_ONLY_PROTECTED_OUTCOMES_UNTOUCHED"
    assert data["target"]["primary_semantics"] == "new_threshold_crossing_within_horizon"
    assert data["target"]["already_active_is_onset_positive"] is False
    protected = data["protected_candidate"]
    assert protected["development_side_may_access_identities"] is False
    assert protected["development_side_may_access_labels"] is False
    assert protected["development_side_may_access_event_counts"] is False
    assert protected["development_side_may_access_episode_durations"] is False
    assert protected["development_side_may_access_scores"] is False


def test_one_physical_positive_episode_has_total_weight_one() -> None:
    data = _load(PREREG)
    rule = data["episode_weight_rule"]
    assert rule["positive_episode_total_weight"] == 1.0
    assert "sum_i w_i(e)=1" in rule["formula"]


def test_shared_bootstrap_draws_are_mandatory() -> None:
    data = _load(PREREG)
    uncertainty = data["uncertainty"]
    assert uncertainty["bootstrap_replicates"] == 10000
    assert uncertainty["bootstrap_unit"] == "complete_physical_sep_episode_or_predeclared_quiet_block"
    assert uncertainty["shared_draws_across_all_models_views_delays_ablations_and_contrasts"] is True
    assert uncertainty["paired"] is True
    assert "independent_bootstrap_redraws_across_compared_conditions" in data["forbidden"]


def test_standard_onset_normalized_and_persistence_views_are_separate() -> None:
    data = _load(PREREG)
    assert data["evaluation_views"] == [
        "WINDOW_OCCURRENCE_STANDARD",
        "NEW_ONSET_CAUSAL",
        "EPISODE_NORMALIZED_ONSET",
        "PROTON_STATE_BLIND_ONSET",
        "PERSISTENCE_DIAGNOSTIC",
    ]
    assert "mixing_already_active_cases_into_onset_skill" in data["forbidden"]


def test_attrition_codes_are_terminal_and_explicit() -> None:
    data = _load(PREREG)
    assert set(data["attrition_terminal_codes"]) == {
        "ELIGIBLE_ONSET_POSITIVE",
        "ELIGIBLE_ONSET_NEGATIVE",
        "ALREADY_ACTIVE_PERSISTENCE",
        "UNRESOLVED_GAP",
        "IMMATURE_OUTCOME_WINDOW",
        "DUPLICATE_OR_AMBIGUOUS",
        "SOURCE_UNAVAILABLE",
        "EXCLUDED_BY_FROZEN_BOUNDARY",
    }


def test_null_result_cannot_trigger_architecture_rescue() -> None:
    data = _load(PREREG)
    assert "retain the null result" in data["hypotheses"]["null_rule"]
    assert data["model_search"]["hyperparameter_search_after_benchmark_delta_inspection"] is False
    assert data["model_search"]["architecture_expansion_in_v1"] is False
    assert "post_result_hyperparameter_rescue" in data["forbidden"]


def test_registry_protects_post_monitor_candidate() -> None:
    registry = _load(REGISTRY)
    candidates = {entry["name"]: entry for entry in registry["protected_candidate_sets"]}
    candidate = candidates["post_monitor_prospective_candidate_after_2025_09_10"]
    assert candidate["status"] == "PROTECTED_NOT_INSPECTED"
    assert candidate["identities_accessed"] is False
    assert candidate["labels_accessed"] is False
    assert candidate["event_counts_accessed"] is False
    assert candidate["model_scores_accessed"] is False
    assert candidate["authorized_for_development"] is False


def test_design_and_current_status_state_the_same_scientific_question() -> None:
    design = DESIGN.read_text(encoding="utf-8").lower()
    current = CURRENT.read_text(encoding="utf-8").lower()
    for phrase in ["episode-normalized", "already-active", "model ranking"]:
        assert phrase in design
        assert phrase in current


def test_design_requires_prediction_level_evidence_and_rank_analysis() -> None:
    data = _load(PREREG)
    required = set(data["required_row_evidence"])
    assert {"issue_timestamp", "eligibility_code", "model_name_version", "probability", "binary_alert"} <= required
    assert data["new_quantities"]["rank_stability_matrix"].startswith("all pairwise model ordering")
    assert "prediction_file_sha256" in data["required_artifact_hashes"]
    assert "bootstrap_draw_tensor_sha256" in data["required_artifact_hashes"]
