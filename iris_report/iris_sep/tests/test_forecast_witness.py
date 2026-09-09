from __future__ import annotations

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

    def comment(self, statement=None, *, delay_seconds=30, edit_seconds=0):
        statement = statement or self.statement()
        created = ISSUE + timedelta(seconds=delay_seconds)
        updated = created + timedelta(seconds=edit_seconds)
        return {
            "id": 12345,
            "created_at": created.isoformat(),
            "updated_at": updated.isoformat(),
            "html_url": "https://github.com/fr3ddykru3g3r/silver-engine/pull/3#issuecomment-12345",
            "body": witness_comment_body(statement),
        }

    def test_server_timestamped_exact_unedited_body_is_accepted(self):
        statement = self.statement()
        receipt = validate_github_comment_witness(
            statement=statement,
            github_comment=self.comment(statement),
        )
        self.assertEqual(receipt["format"], "IRIS_SEP_GITHUB_FORECAST_WITNESS_V2")
        self.assertTrue(receipt["pre_outcome_digest_existence_witnessed"])
        self.assertTrue(receipt["github_comment_unedited_since_creation"])
        self.assertFalse(receipt["independence_verified"])
        self.assertEqual(receipt["witness_delay_seconds"], 30.0)
        self.assertEqual(receipt["github_comment_created_at"], receipt["github_comment_updated_at"])

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

    def test_comment_edited_after_outcome_is_rejected_even_if_body_matches(self):
        statement = self.statement()
        comment = self.comment(statement, edit_seconds=24 * 60 * 60 + 60)
        with self.assertRaisesRegex(ForecastWitnessError, "edited after creation"):
            validate_github_comment_witness(statement=statement, github_comment=comment)

    def test_comment_edited_inside_five_minute_window_is_still_rejected(self):
        statement = self.statement()
        comment = self.comment(statement, edit_seconds=10)
        with self.assertRaisesRegex(ForecastWitnessError, "edited after creation"):
            validate_github_comment_witness(statement=statement, github_comment=comment)

    def test_edit_and_restore_body_is_rejected_by_updated_timestamp(self):
        statement = self.statement()
        comment = self.comment(statement, edit_seconds=1)
        # The final body is byte-for-byte correct, but the server timestamp proves
        # that the comment was edited after creation.
        self.assertEqual(comment["body"], witness_comment_body(statement))
        with self.assertRaisesRegex(ForecastWitnessError, "edited after creation"):
            validate_github_comment_witness(statement=statement, github_comment=comment)

    def test_missing_updated_at_is_rejected(self):
        statement = self.statement()
        comment = self.comment(statement)
        del comment["updated_at"]
        with self.assertRaisesRegex(ForecastWitnessError, "updated_at"):
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
