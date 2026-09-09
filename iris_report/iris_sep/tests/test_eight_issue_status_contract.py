from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "config" / "eight_issue_status_2026-09-08.json"
CURRENT = ROOT / "CURRENT_STATUS.md"


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    payload = f"blob {len(data)}\0".encode("ascii") + data
    return hashlib.sha1(payload).hexdigest()


def test_eight_issue_status_has_no_in_progress_items() -> None:
    data = json.loads(STATUS.read_text(encoding="utf-8"))
    assert data["format"] == "IRIS_SEP_EIGHT_ISSUE_STATUS_V3"
    assert data["status_date"] == "2026-09-08"
    assert len(data["items"]) == 8
    for key, item in data["items"].items():
        status = str(item["status"])
        assert "IN_PROGRESS" not in status, key


def test_item8_references_existing_hash_bound_evidence_and_visual_files() -> None:
    data = json.loads(STATUS.read_text(encoding="utf-8"))
    item = data["items"]["8_evidence_and_paper_generation"]
    assert item["status"] == "PASS_INTERNAL_EVIDENCE_PACKAGE"
    evidence = item["evidence"]

    pairs = [
        (evidence["evidence_dossier"], evidence["evidence_dossier_git_blob_sha1"]),
        (evidence["student_writing_template"], evidence["student_writing_template_git_blob_sha1"]),
        (evidence["judge_result_tables"], evidence["judge_result_tables_git_blob_sha1"]),
    ]
    pairs.extend((entry["path"], entry["git_blob_sha1"]) for entry in evidence["judge_figures"])

    assert len(pairs) == 6
    for relative, expected_sha in pairs:
        path = ROOT / relative
        assert path.is_file(), relative
        assert _git_blob_sha1(path) == expected_sha, relative

    assert evidence["final_submission_prose_ai_generated"] is False


def test_judge_figures_surface_the_frozen_evidence_boundaries() -> None:
    data = json.loads(STATUS.read_text(encoding="utf-8"))
    figures = data["items"]["8_evidence_and_paper_generation"]["evidence"]["judge_figures"]
    texts = [(ROOT / entry["path"]).read_text(encoding="utf-8") for entry in figures]

    assert "ABSTAIN" in texts[0]
    assert "NO ALERT" in texts[0]
    assert "−28.5% FP, −1 TP" in texts[1]
    assert "−25.4% FP, −3 TP" in texts[1]
    assert "CMASKL" in texts[2]
    assert "MEANGBL" in texts[2]
    assert "USFLUXL" in texts[2]
    assert "18 positions" in texts[2]
    assert "NO PROSPECTIVE SKILL FORECAST EMITTED" in texts[2]


def test_current_status_is_current_and_matches_primary_disposition() -> None:
    text = CURRENT.read_text(encoding="utf-8")
    assert "**Status date:** 2026-09-08" in text
    assert "RETROSPECTIVE" in text.upper()
    assert "CMASKL" in text
    assert "MEANGBL" in text
    assert "USFLUXL" in text
    assert "18 frozen feature-vector positions" in text
    assert "no forecast probability" in text.lower()
    assert "Not yet complete:" not in text
    assert "reloadable exported package" not in text


def test_alert_filter_disposition_is_frozen_to_plain_v3() -> None:
    data = json.loads(STATUS.read_text(encoding="utf-8"))
    decision = data["alert_filter_decision"]
    assert decision["status"] == "REJECT_ADDITIONAL_POST_HOC_FILTERING_KEEP_PLAIN_V3"
    assert decision["frozen_candidate"] == "IRIS_AVAILABILITY_DISTILLED_EVIDENCE_STACK_V3"
    assert decision["failed_decision_filter_run"] == 34206898291
    assert decision["monotone_veto_run"] == 34216432375

    no_xrs = decision["monotone_veto_score"]["NO_XRS"]
    assert (no_xrs["v3_tp"], no_xrs["v3_fp"]) == (13, 603)
    assert (no_xrs["veto_tp"], no_xrs["veto_fp"]) == (12, 431)
    assert no_xrs["true_positive_change"] == -1

    no_proton = decision["monotone_veto_score"]["NO_PROTON"]
    assert (no_proton["v3_tp"], no_proton["v3_fp"]) == (16, 796)
    assert (no_proton["veto_tp"], no_proton["veto_fp"]) == (13, 594)
    assert no_proton["true_positive_change"] == -3


def test_status_does_not_claim_prospective_skill_or_award_outcome() -> None:
    data = json.loads(STATUS.read_text(encoding="utf-8"))
    assert data["independent_prospective_skill_established"] is False
    assert data["award_outcome_claimed"] is False
