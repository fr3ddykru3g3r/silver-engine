import unittest

from iris_sep.forecast_validity import build_validity_record


class ForecastValidityContractTests(unittest.TestCase):
    def base(self):
        return {
            "issue_time": "2026-09-12T00:00:00Z",
            "alert": True,
            "input_age_seconds": {"proton": 60},
            "source_hashes": {"proton": "a" * 64},
            "model_hash": "b" * 64,
            "threshold_hash": "c" * 64,
            "ledger_previous_hash": None,
            "ledger_is_genesis": True,
            "observation_times": {"proton": "2026-09-11T23:59:00Z"},
            "flags": {},
            "protected_outcomes_accessed": False,
        }

    def test_clean_record_is_valid(self):
        row = build_validity_record(self.base())
        self.assertEqual(row["validity_state"], "VALID")
        self.assertIs(row["alert"], True)
        self.assertEqual(row["reason_codes"], [])
        self.assertIs(row["protected_outcomes_accessed"], False)

    def test_future_observation_forces_abstain(self):
        data = self.base()
        data["observation_times"]["proton"] = "2026-09-12T00:00:01Z"
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "ABSTAIN")
        self.assertIn("FUTURE_OBSERVATION", row["reason_codes"])
        self.assertIsNone(row["alert"])

    def test_stale_input_forces_abstain(self):
        data = self.base()
        data["flags"] = {"stale_input": True}
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "ABSTAIN")
        self.assertIn("STALE_INPUT", row["reason_codes"])

    def test_structural_absence_is_never_recovered(self):
        data = self.base()
        data["flags"] = {"structural_unavailability": True, "transient_forward_fill_applied": True}
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "ABSTAIN")
        self.assertIn("STRUCTURAL_UNAVAILABILITY", row["reason_codes"])
        self.assertIsNone(row["alert"])

    def test_ambiguous_proton_channel_forces_abstain(self):
        data = self.base()
        data["flags"] = {"ambiguous_proton_channel": True}
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "ABSTAIN")
        self.assertIn("AMBIGUOUS_PROTON_CHANNEL", row["reason_codes"])

    def test_ledger_integrity_failure_forces_abstain(self):
        data = self.base()
        data["ledger_is_genesis"] = False
        data["ledger_previous_hash"] = "tampered"
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "ABSTAIN")
        self.assertIn("LEDGER_INTEGRITY_FAILURE", row["reason_codes"])

    def test_causal_forward_fill_within_external_envelope_is_degraded(self):
        data = self.base()
        data["flags"] = {"transient_forward_fill_applied": True}
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "DEGRADED")
        self.assertEqual(row["reason_codes"], ["TRANSIENT_FORWARD_FILL_APPLIED"])
        self.assertIs(row["alert"], True)

    def test_excessive_transient_loss_overrides_degraded(self):
        data = self.base()
        data["flags"] = {
            "transient_forward_fill_applied": True,
            "excessive_transient_loss": True,
        }
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "ABSTAIN")
        self.assertIn("EXCESSIVE_TRANSIENT_LOSS", row["reason_codes"])
        self.assertIsNone(row["alert"])

    def test_failed_causal_receipt_forces_abstain(self):
        data = self.base()
        data["flags"] = {"causal_availability_receipt_failed": True}
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "ABSTAIN")
        self.assertIn("CAUSAL_AVAILABILITY_RECEIPT_FAILED", row["reason_codes"])

    def test_uncertain_provenance_forces_abstain(self):
        data = self.base()
        data["flags"] = {"provenance_uncertain": True}
        row = build_validity_record(data)
        self.assertEqual(row["validity_state"], "ABSTAIN")
        self.assertIn("PROVENANCE_UNCERTAIN", row["reason_codes"])


if __name__ == "__main__":
    unittest.main()
