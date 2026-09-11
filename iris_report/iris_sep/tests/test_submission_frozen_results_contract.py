"""Submission-facing regression guards for the frozen IRIS-SEP scientific record.

These tests do not regenerate data-dependent science. They prevent later documentation or
submission packaging edits from silently drifting away from the immutable replay receipts.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "submission" / "IRIS_2026_EVIDENCE_BUNDLE_V1"
FROZEN = json.loads((BUNDLE / "FROZEN_RESULTS.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))
STATUS = (ROOT / "CURRENT_STATUS.md").read_text(encoding="utf-8")
RECEIPT = (
    ROOT
    / "architecture"
    / "FINAL_TECHNICAL_RESULTS_AND_FREEZE_RECEIPT_2026-09-11.md"
).read_text(encoding="utf-8")


def test_frozen_artifact_hashes_and_protected_boundary():
    assert FROZEN["protected_post_2025_outcomes_accessed"] is False
    assert (
        FROZEN["immutable_artifacts"]["model_free_confirmation"]["archive_sha256"]
        == "4d75cb08d9beb8ac1761e138ffcf0464da9456bd7111e5318e83a8d042314f4f"
    )
    assert (
        FROZEN["immutable_artifacts"]["fixed_model_replay"]["archive_sha256"]
        == "81ec33ee08c89d0e4627e6761fbfa993eccf29e436b6ac32efbfc6f7d954c9b0"
    )
    assert FROZEN["prospective_confirmation"]["status"] == "FROZEN_NOT_EXECUTED"
    assert FROZEN["prospective_confirmation"]["development_side_outcome_access"] == "FORBIDDEN"


def test_denominator_guard_is_machine_consistent():
    audit = FROZEN["model_free_audit"]
    replay = FROZEN["fixed_replay"]
    guard = MANIFEST["denominator_guard"]
    assert audit["new_onset_windows"] == 228
    assert replay["matched_onset_episode_units"] == 85
    assert replay["quiet_block_units"] == 1080
    assert guard["full_table_onset_windows"] == audit["new_onset_windows"]
    assert guard["matched_replay_distinct_onset_episodes"] == replay["matched_onset_episode_units"]
    assert guard["matched_replay_quiet_blocks"] == replay["quiet_block_units"]
    assert audit["new_onset_windows"] != replay["matched_onset_episode_units"]


def test_headline_tss_rounds_to_submission_manifest():
    tss = FROZEN["fixed_replay"]["matched_tss"]["xgb_joint"]
    headline = MANIFEST["headline_metrics"]
    assert round(tss["mapped_occurrence"], 3) == headline["joint_xgboost_tss_mapped_occurrence"]
    assert round(tss["episode_normalized_occurrence"], 3) == headline["joint_xgboost_tss_episode_normalized"]
    assert round(tss["new_onset"], 3) == headline["joint_xgboost_tss_new_onset"]
    assert "0.726 -> 0.621 -> 0.437" in STATUS
    assert "0.726455 -> 0.621128 -> 0.437234" in RECEIPT


def test_operating_characteristics_match_submission_manifest():
    oc = FROZEN["fixed_replay"]["xgb_joint_new_onset_operating_characteristics"]
    headline = MANIFEST["headline_metrics"]
    assert oc["tp"] == 41
    assert oc["fn"] == 44
    assert oc["fp"] == 330
    assert oc["tn"] == 6984
    assert oc["sensitivity_percent"] == headline["joint_xgboost_onset_sensitivity_percent"]
    assert oc["false_alarm_ratio_percent"] == headline["joint_xgboost_onset_false_alarm_ratio_percent"]
    assert oc["false_positive_rate_percent"] == headline["joint_xgboost_onset_false_positive_rate_percent"]
    assert "88.95%" in STATUS and "4.51%" in STATUS


def test_primary_bootstrap_intervals_remain_strictly_negative():
    contrasts = FROZEN["fixed_replay"]["primary_bootstrap_contrasts"]
    for result in contrasts.values():
        lo, hi = result["ci95_percentile"]
        assert lo < 0.0
        assert hi < 0.0
        assert result["all_10000_draws_negative"] is True


def test_secondary_xgb_contrast_does_not_claim_reliable_superiority():
    secondary = FROZEN["fixed_replay"]["secondary_bootstrap_contrasts"][
        "xgb_joint_minus_xgb_no_proton_onset"
    ]
    lo, hi = secondary["ci95_percentile"]
    assert lo < 0.0 < hi
    assert secondary["reliable_rank_reversal_supported"] is False
    assert round(secondary["point_change"], 3) == MANIFEST["headline_metrics"][
        "joint_minus_proton_free_onset_tss_contrast"
    ]
    expected_interval = MANIFEST["headline_metrics"]["joint_minus_proton_free_onset_tss_interval"]
    assert [round(lo, 3), round(hi, 3)] == expected_interval


def test_receipt_keeps_historical_and_prospective_evidence_separate():
    assert "RETROSPECTIVE_TECHNICAL_STACK_FROZEN" in RECEIPT
    assert "FROZEN_NOT_EXECUTED" in RECEIPT
    assert "not operational readiness" in RECEIPT.lower() or "operational-readiness" in RECEIPT.lower()
    assert FROZEN["historical_evidence_class"] == "DEVELOPMENT_EXPOSED_PUBLIC_HISTORICAL_DATA"
