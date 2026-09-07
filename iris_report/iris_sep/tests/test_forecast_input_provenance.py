from __future__ import annotations

import unittest
from datetime import datetime, timezone

from iris_report.iris_sep.src.iris_sep.forecast_input_provenance import (
    ForecastInputProvenanceError,
    evaluate_forecast_input_provenance,
    require_forecast_input_provenance,
)


ISSUED = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)
DEV_CUTOFF = datetime(2026, 8, 31, 23, 59, tzinfo=timezone.utc)
REQUIRED = {
    "XRS": ("xrs_short",),
    "PROTON": ("proton_gt10",),
    "MAGNETIC_SHARP_SMARP": ("sharp_r_value",),
}


def native(feature_name, family, *, state="NATIVE_OBSERVED"):
    return {
        "feature_name": feature_name,
        "family": family,
        "state": state,
        "source_revision": "fixture-v1",
        "observed_at_utc": "2026-09-07T11:50:00Z",
        "published_at_utc": "2026-09-07T11:55:00Z",
        "uses_future_values": False,
        "retrospective_fit": False,
        "transform_fit_through_utc": None,
    }


def valid_records():
    return [
        native("xrs_short", "XRS"),
        native("proton_gt10", "PROTON", state="ALTERNATE_SOURCE_OBSERVED"),
        {
            "feature_name": "sharp_r_value",
            "family": "MAGNETIC_SHARP_SMARP",
            "state": "RECONSTRUCTED_CAUSAL",
            "source_revision": "fixture-v1",
            "observed_at_utc": "2026-09-07T11:00:00Z",
            "published_at_utc": "2026-09-07T11:10:00Z",
            "uses_future_values": False,
            "retrospective_fit": False,
            "transform_id": "past-only-fixture-v1",
            "transform_fit_through_utc": "2026-08-31T23:59:00Z",
        },
    ]


class ForecastInputProvenanceTests(unittest.TestCase):
    def test_valid_native_alternate_and_causal_reconstruction_pass(self):
        result = require_forecast_input_provenance(
            issued_at=ISSUED,
            records=valid_records(),
            required_features=REQUIRED,
            development_information_cutoff=DEV_CUTOFF,
        )
        self.assertEqual(result["status"], "VALID")
        self.assertTrue(result["forecast_probability_permitted"])
        self.assertEqual(result["required_feature_count"], 3)
        self.assertEqual(result["lineage_record_count"], 3)
        self.assertTrue(all(value["permitted"] for value in result["family_results"].values()))
        self.assertEqual(len(result["manifest_sha256"]), 64)

    def test_missing_lineage_fails_closed(self):
        records = valid_records()[:-1]
        result = evaluate_forecast_input_provenance(
            issued_at=ISSUED,
            records=records,
            required_features=REQUIRED,
            development_information_cutoff=DEV_CUTOFF,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["forecast_probability_permitted"])
        self.assertIn("MAGNETIC_SHARP_SMARP/sharp_r_value:MISSING_LINEAGE", result["reasons"])

    def test_unknown_aggregate_lineage_is_blocked_even_when_value_is_finite(self):
        records = valid_records()
        records[1] = native("proton_gt10", "PROTON", state="UNKNOWN")
        result = evaluate_forecast_input_provenance(
            issued_at=ISSUED,
            records=records,
            required_features=REQUIRED,
            development_information_cutoff=DEV_CUTOFF,
        )
        self.assertFalse(result["forecast_probability_permitted"])
        self.assertTrue(any("NONCAUSAL_OR_UNUSABLE_STATE:UNKNOWN" in reason for reason in result["reasons"]))

    def test_noncausal_retrospective_reconstruction_is_blocked(self):
        records = valid_records()
        records[2]["state"] = "RECONSTRUCTED_NONCAUSAL_OR_RETROSPECTIVE"
        records[2]["uses_future_values"] = True
        records[2]["retrospective_fit"] = True
        result = evaluate_forecast_input_provenance(
            issued_at=ISSUED,
            records=records,
            required_features=REQUIRED,
            development_information_cutoff=DEV_CUTOFF,
        )
        self.assertFalse(result["forecast_probability_permitted"])
        joined = "\n".join(result["reasons"])
        self.assertIn("RECONSTRUCTED_NONCAUSAL_OR_RETROSPECTIVE", joined)
        self.assertIn("FUTURE_VALUE_USE_NOT_EXPLICITLY_FALSE", joined)
        self.assertIn("RETROSPECTIVE_FIT_NOT_EXPLICITLY_FALSE", joined)

    def test_future_publication_is_blocked(self):
        records = valid_records()
        records[0]["published_at_utc"] = "2026-09-07T12:01:00Z"
        result = evaluate_forecast_input_provenance(
            issued_at=ISSUED,
            records=records,
            required_features=REQUIRED,
            development_information_cutoff=DEV_CUTOFF,
        )
        self.assertTrue(any("PUBLICATION_AFTER_ISSUE_TIME" in reason for reason in result["reasons"]))

    def test_future_nearest_style_recovery_is_blocked(self):
        records = valid_records()
        records[0]["uses_future_values"] = True
        result = evaluate_forecast_input_provenance(
            issued_at=ISSUED,
            records=records,
            required_features=REQUIRED,
            development_information_cutoff=DEV_CUTOFF,
        )
        self.assertTrue(any("FUTURE_VALUE_USE_NOT_EXPLICITLY_FALSE" in reason for reason in result["reasons"]))

    def test_transform_fit_after_frozen_development_cutoff_is_blocked(self):
        records = valid_records()
        records[2]["transform_fit_through_utc"] = "2026-09-01T00:00:00Z"
        result = evaluate_forecast_input_provenance(
            issued_at=ISSUED,
            records=records,
            required_features=REQUIRED,
            development_information_cutoff=DEV_CUTOFF,
        )
        self.assertTrue(any("TRANSFORM_FIT_EXTENDS_BEYOND_DEVELOPMENT_CUTOFF" in reason for reason in result["reasons"]))

    def test_reconstruction_requires_declared_development_cutoff(self):
        result = evaluate_forecast_input_provenance(
            issued_at=ISSUED,
            records=valid_records(),
            required_features=REQUIRED,
        )
        self.assertTrue(any("DEVELOPMENT_INFORMATION_CUTOFF_REQUIRED_FOR_RECONSTRUCTION" in reason for reason in result["reasons"]))

    def test_duplicate_lineage_is_blocked(self):
        records = valid_records() + [native("xrs_short", "XRS")]
        result = evaluate_forecast_input_provenance(
            issued_at=ISSUED,
            records=records,
            required_features=REQUIRED,
            development_information_cutoff=DEV_CUTOFF,
        )
        self.assertIn("XRS/xrs_short:DUPLICATE_LINEAGE", result["reasons"])

    def test_require_raises_with_reasons(self):
        records = valid_records()
        records[0]["state"] = "UNKNOWN"
        with self.assertRaises(ForecastInputProvenanceError) as context:
            require_forecast_input_provenance(
                issued_at=ISSUED,
                records=records,
                required_features=REQUIRED,
                development_information_cutoff=DEV_CUTOFF,
            )
        self.assertIn("prospective forecast input blocked", str(context.exception))


if __name__ == "__main__":
    unittest.main()
