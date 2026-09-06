"""Dependency-free synthetic tests for the Luna D identity contract.

No AARP NetCDF, HMI FITS, SEP labels, SEPVAL rows, or locked-test artifact is
read by these tests.
"""

from __future__ import annotations

import unittest

from .identity_bridge import (
    build_bridge_receipt,
    build_no_crosswalk_result,
    evaluate_bridge,
)


def _provenance(prefix: str, row_id: str, role: str | None = None) -> dict[str, str]:
    out = {
        "source_uri": f"synthetic://{prefix}",
        "source_version": "toy-v1",
        "source_sha256": prefix[0] * 64,
        "source_row_id": row_id,
        "retrieved_at_utc": "2026-09-04T00:00:00Z",
    }
    if role is not None:
        out.update({"artifact_role": role, "row_sha256": prefix[0] * 64})
    return out


def _aarp(source_id: str = "2020.01.01_00:00:00_7h@1h_101.fits") -> dict:
    return {
        "aarp_source_id": source_id,
        "aarp_number": 101,
        "issue_time_utc": "2020-01-01T00:00:00Z",
        "provenance": _provenance("a", "aarp-row-1"),
    }


def _hmi(harpnum: int = 501, t_rec_tai: str = "2020.01.01_00:00:00_TAI") -> dict:
    return {
        "harpnum": harpnum,
        "t_rec_tai": t_rec_tai,
        "hmi_record_id": f"hmi-{harpnum}",
        "provenance": _provenance("b", "hmi-row-1"),
    }


def _candidate() -> dict:
    return {
        "mapping_id": "map-1",
        "aarp_source_id": "2020.01.01_00:00:00_7h@1h_101.fits",
        "aarp_number": 101,
        "harpnum": 501,
        "t_rec_tai": "2020.01.01_00:00:00_TAI",
        "mapping_method": "authoritative_crosswalk",
        "exact_key_fields": ["aarp_source_id", "harpnum", "t_rec_tai"],
        "ambiguity_status": "unambiguous",
        "candidate_count": 1,
        "authority_provenance": _provenance("c", "crosswalk-row-1", "identity_crosswalk"),
        "time_proof": {
            "kind": "authoritative_exact_observation",
            "aarp_issue_time_utc": "2020-01-01T00:00:00Z",
            "hmi_t_rec_tai": "2020.01.01_00:00:00_TAI",
            "authority_row_id": "crosswalk-row-1",
        },
        "region_proof": {
            "kind": "authoritative_region_crosswalk",
            "aarp_number": 101,
            "harpnum": 501,
            "authority_row_id": "crosswalk-row-1",
        },
    }


class IdentityBridgeTests(unittest.TestCase):
    def test_authoritative_exact_crosswalk_is_accepted_with_provenance(self) -> None:
        result = evaluate_bridge([_aarp()], [_hmi()], [_candidate()])
        self.assertEqual(result.as_dict()["status"], "PASS")
        self.assertTrue(result.as_dict()["fusion_allowed"])
        mapping = result.as_dict()["accepted_mappings"][0]
        self.assertEqual(mapping["hmi_key"]["harpnum"], 501)
        self.assertEqual(mapping["hmi_key"]["t_rec_tai"], "2020.01.01_00:00:00_TAI")
        self.assertIn("authority", mapping["provenance"])
        self.assertEqual(mapping["mapping_method"], "authoritative_crosswalk")

    def test_no_crosswalk_is_fail_closed(self) -> None:
        result = build_no_crosswalk_result()
        self.assertEqual(result.as_dict()["status"], "REJECT")
        self.assertFalse(result.as_dict()["fusion_allowed"])
        self.assertEqual(result.as_dict()["rejections"][0]["reason_code"], "NO_AUTHORITATIVE_CROSSWALK")

    def test_missing_crosswalk_row_keeps_aia_external_only(self) -> None:
        result = evaluate_bridge([_aarp()], [_hmi()], [])
        self.assertEqual(result.as_dict()["status"], "REJECT")
        self.assertEqual(result.as_dict()["rejections"][0]["reason_code"], "NO_AUTHORITATIVE_CROSSWALK")

    def test_date_nearest_and_numeric_equality_methods_are_rejected(self) -> None:
        for method in ("date", "nearest_timestamp", "numeric_equality"):
            candidate = _candidate()
            candidate["mapping_method"] = method
            result = evaluate_bridge([_aarp()], [_hmi()], [candidate])
            self.assertEqual(result.as_dict()["status"], "REJECT", method)
            self.assertEqual(
                result.as_dict()["rejections"][0]["reason_code"],
                "NON_AUTHORITATIVE_MAPPING_METHOD",
                method,
            )

    def test_two_candidates_for_one_aarp_are_ambiguous(self) -> None:
        second = _candidate()
        second["mapping_id"] = "map-2"
        second["harpnum"] = 502
        second["t_rec_tai"] = "2020.01.01_01:00:00_TAI"
        result = evaluate_bridge([_aarp()], [_hmi(), _hmi(502, second["t_rec_tai"])], [_candidate(), second])
        self.assertFalse(result.accepted)
        self.assertEqual(result.as_dict()["rejections"][0]["reason_code"], "AMBIGUOUS_CROSSWALK")

    def test_one_hmi_key_cannot_be_reused(self) -> None:
        second_aarp = _aarp("2020.01.02_00:00:00_7h@1h_102.fits")
        second_aarp["aarp_number"] = 102
        second_aarp["issue_time_utc"] = "2020-01-02T00:00:00Z"
        second = _candidate()
        second["mapping_id"] = "map-2"
        second["aarp_source_id"] = second_aarp["aarp_source_id"]
        second["aarp_number"] = 102
        second["time_proof"]["aarp_issue_time_utc"] = second_aarp["issue_time_utc"]
        second["region_proof"]["aarp_number"] = 102
        result = evaluate_bridge([_aarp(), second_aarp], [_hmi()], [_candidate(), second])
        self.assertEqual(len(result.accepted), 1)
        self.assertIn("HMI_KEY_REUSED", {item.reason_code for item in result.rejected})

    def test_outcome_fields_and_locked_test_are_rejected(self) -> None:
        row = _aarp()
        row["Future_OSEP_label"] = 0
        result = evaluate_bridge([row], [_hmi()], [])
        self.assertEqual(result.as_dict()["status"], "REJECT")
        locked = _aarp()
        locked["partition"] = "locked_test"
        result = evaluate_bridge([locked], [_hmi()], [])
        self.assertEqual(result.as_dict()["status"], "REJECT")

    def test_receipt_is_hashed_and_declares_rules(self) -> None:
        result = evaluate_bridge([_aarp()], [_hmi()], [_candidate()])
        receipt = build_bridge_receipt(result, generated_at_utc="2026-09-04T01:00:00Z")
        self.assertEqual(receipt["status"], "PASS")
        self.assertEqual(len(receipt["receipt_sha256"]), 64)
        self.assertFalse(receipt["scientific_boundaries"]["locked_test_outcomes_accessed"])
        self.assertEqual(receipt["scientific_boundaries"]["date_only_join"], "REJECTED")


if __name__ == "__main__":
    unittest.main()
