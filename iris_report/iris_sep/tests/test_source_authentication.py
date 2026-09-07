from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from iris_report.iris_sep.src.iris_sep.source_authentication import (
    authenticate_acquisition_receipts,
    build_acquisition_receipt,
    require_authenticated_acquisitions,
    SourceAuthenticationError,
)


ISSUE = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
DUMMY_SHA = "a" * 64


class SourceAuthenticationTests(unittest.TestCase):
    def swpc_receipt(self, *, retrieved_at=ISSUE - timedelta(minutes=1), url=None):
        return build_acquisition_receipt(
            source_id="NOAA_SWPC_PRIMARY_XRS_7D",
            retrieved_at=retrieved_at,
            artifact_sha256=DUMMY_SHA,
            artifact_bytes=123,
            source_url=url or "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
            observation_first_utc="2026-09-06T12:00:00Z",
            observation_last_utc="2026-09-07T11:59:00Z",
        )

    def test_valid_official_snapshot_passes(self):
        result = require_authenticated_acquisitions(
            issue_time=ISSUE,
            receipts=[self.swpc_receipt()],
            required_source_ids=["NOAA_SWPC_PRIMARY_XRS_7D"],
        )
        self.assertTrue(result["authenticated_for_prospective_use"])
        self.assertEqual(result["source_results"][0]["reasons"], [])

    def test_wrong_host_fails_closed(self):
        receipt = self.swpc_receipt(url="https://example.com/json/goes/primary/xrays-7-day.json")
        result = authenticate_acquisition_receipts(
            issue_time=ISSUE,
            receipts=[receipt],
            required_source_ids=["NOAA_SWPC_PRIMARY_XRS_7D"],
        )
        self.assertFalse(result["authenticated_for_prospective_use"])
        self.assertTrue(any("SOURCE_HOST_MISMATCH" in reason for reason in result["reasons"]))

    def test_retrieval_after_issue_is_not_prospective_evidence(self):
        receipt = self.swpc_receipt(retrieved_at=ISSUE + timedelta(seconds=1))
        result = authenticate_acquisition_receipts(
            issue_time=ISSUE,
            receipts=[receipt],
            required_source_ids=["NOAA_SWPC_PRIMARY_XRS_7D"],
        )
        self.assertFalse(result["authenticated_for_prospective_use"])
        self.assertTrue(any("RETRIEVED_AFTER_FORECAST_ISSUE" in reason for reason in result["reasons"]))

    def test_registry_digest_tamper_is_detected(self):
        receipt = self.swpc_receipt()
        receipt["registry_sha256"] = "b" * 64
        result = authenticate_acquisition_receipts(
            issue_time=ISSUE,
            receipts=[receipt],
            required_source_ids=["NOAA_SWPC_PRIMARY_XRS_7D"],
        )
        self.assertFalse(result["authenticated_for_prospective_use"])
        joined = "\n".join(result["reasons"])
        self.assertIn("REGISTRY_DIGEST_MISMATCH", joined)
        self.assertIn("RECEIPT_DIGEST_MISMATCH", joined)

    def test_drms_query_must_bind_registered_series(self):
        receipt = build_acquisition_receipt(
            source_id="JSOC_HMI_SHARP_NRT",
            retrieved_at=ISSUE - timedelta(minutes=1),
            artifact_sha256=DUMMY_SHA,
            artifact_bytes=42,
            query_identity="hmi.sharp_720s[][2026.09.07_00:00:00_TAI/1d]",
        )
        result = authenticate_acquisition_receipts(
            issue_time=ISSUE,
            receipts=[receipt],
            required_source_ids=["JSOC_HMI_SHARP_NRT"],
        )
        self.assertFalse(result["authenticated_for_prospective_use"])
        self.assertTrue(any("DRMS_SERIES_NOT_BOUND_IN_QUERY" in reason for reason in result["reasons"]))

    def test_missing_required_source_is_blocked(self):
        with self.assertRaises(SourceAuthenticationError):
            require_authenticated_acquisitions(
                issue_time=ISSUE,
                receipts=[self.swpc_receipt()],
                required_source_ids=["NOAA_SWPC_PRIMARY_XRS_7D", "NOAA_SWPC_PRIMARY_PROTON_7D"],
            )


if __name__ == "__main__":
    unittest.main()
