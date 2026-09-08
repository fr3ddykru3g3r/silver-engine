from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from iris_report.iris_sep.src.iris_sep.forecast_witness import (
    ForecastWitnessError,
    build_witness_statement,
    validate_github_comment_witness,
    witness_comment_body,
)


ISSUE = datetime(2026, 9, 8, 8, 0, tzinfo=timezone.utc)


class ForecastWitnessTests(unittest.TestCase):
    def statement(self):
        return build_witness_statement(
            issued_at=ISSUE,
            forecast_seal_sha256="a" * 64,
            package_manifest_sha256="b" * 64,
            comparison_seal_sha256="c" * 64,
        )

    def comment(self, statement=None, *, delay_seconds=30):
        statement = statement or self.statement()
        return {
            "id": 12345,
            "created_at": (ISSUE + timedelta(seconds=delay_seconds)).isoformat(),
            "html_url": "https://github.com/fr3ddykru3g3r/silver-engine/pull/3#issuecomment-12345",
            "body": witness_comment_body(statement),
        }

    def test_server_timestamped_exact_body_is_accepted(self):
        statement = self.statement()
        receipt = validate_github_comment_witness(
            statement=statement,
            github_comment=self.comment(statement),
        )
        self.assertTrue(receipt["pre_outcome_digest_existence_witnessed"])
        self.assertFalse(receipt["independence_verified"])
        self.assertEqual(receipt["witness_delay_seconds"], 30.0)

    def test_late_server_witness_is_rejected(self):
        with self.assertRaisesRegex(ForecastWitnessError, "more than five minutes"):
            validate_github_comment_witness(
                statement=self.statement(),
                github_comment=self.comment(delay_seconds=301),
            )

    def test_body_tamper_is_rejected(self):
        statement = self.statement()
        comment = self.comment(statement)
        comment["body"] += "tamper"
        with self.assertRaisesRegex(ForecastWitnessError, "does not exactly match"):
            validate_github_comment_witness(statement=statement, github_comment=comment)

    def test_rehashed_statement_substitution_changes_required_body(self):
        statement = self.statement()
        original_comment = self.comment(statement)
        changed = build_witness_statement(
            issued_at=ISSUE,
            forecast_seal_sha256="d" * 64,
            package_manifest_sha256="b" * 64,
            comparison_seal_sha256="c" * 64,
        )
        with self.assertRaisesRegex(ForecastWitnessError, "does not exactly match"):
            validate_github_comment_witness(statement=changed, github_comment=original_comment)

    def test_non_github_server_record_is_rejected(self):
        statement = self.statement()
        comment = self.comment(statement)
        comment["html_url"] = "https://example.com/comment/12345"
        with self.assertRaisesRegex(ForecastWitnessError, "github.com"):
            validate_github_comment_witness(statement=statement, github_comment=comment)


if __name__ == "__main__":
    unittest.main()
