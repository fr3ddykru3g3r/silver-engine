from copy import deepcopy
from datetime import timedelta
import unittest
from iris_report.iris_sep.tests.test_sealed_evaluation import BASE, seal, complete_series
from iris_report.iris_sep.src.iris_sep import sealed_evaluation as se
from iris_report.iris_sep.src.iris_sep.sealed_comparison import evaluate_sealed_comparators, SealedComparisonError


class OutcomeIntegrityTests(unittest.TestCase):
    def labels(self, times, flux):
        return se.derive_new_crossing_labels(forecast_seals=[seal(BASE)], proton_times=times, proton_flux=flux)

    def test_five_minutes_cannot_label_twenty_four_hours_negative(self):
        result = self.labels([BASE-timedelta(minutes=5), BASE+timedelta(minutes=5)], [1, 1])
        self.assertIsNone(result['rows'][0]['label'])
        scored = se.evaluate_sealed_cohort(forecast_seals=[seal(BASE)], label_receipt=result)
        self.assertEqual(scored['eligible_rows'], 0)
        self.assertEqual(scored['unresolved_outcome_rows'], 1)

    def test_complete_positive_and_negative_windows(self):
        for positives, expected in [([], 0), ([BASE], 1)]:
            times, flux = complete_series([BASE], positives)
            self.assertEqual(self.labels(times, flux)['rows'][0]['label'], expected)

    def test_gap_invalid_flux_stale_issue_and_missing_endpoint_are_unknown(self):
        times, flux = complete_series([BASE], [])
        cases = [(times[:-1], flux[:-1]), (times[3:], flux[3:]),
                 (times[:10]+times[12:], flux[:10]+flux[12:])]
        for invalid in [float('nan'), -1, float('inf')]:
            changed = flux.copy(); changed[10] = invalid; cases.append((times, changed))
        for tt, ff in cases:
            with self.subTest(samples=len(tt), invalid=ff[10] if len(ff)>10 else None):
                self.assertIsNone(self.labels(tt, ff)['rows'][0]['label'])

    def test_duplicate_samples_rejected(self):
        with self.assertRaisesRegex(se.SealedEvaluationError, 'duplicate proton'):
            self.labels([BASE, BASE], [1, 12])

    def test_label_tampering_rejected_by_both_evaluators(self):
        times, flux = complete_series([BASE], [])
        labels = self.labels(times, flux); labels['rows'][0]['label'] = 1
        with self.assertRaisesRegex(se.SealedEvaluationError, 'digest mismatch'):
            se.evaluate_sealed_cohort(forecast_seals=[seal(BASE)], label_receipt=labels)
        with self.assertRaisesRegex(SealedComparisonError, 'digest mismatch'):
            evaluate_sealed_comparators(comparison_receipts=[], label_receipt=labels)

    def test_duplicate_forecasts_cannot_inflate_support(self):
        times, flux = complete_series([BASE], [BASE]); labels = self.labels(times, flux)
        with self.assertRaisesRegex(se.SealedEvaluationError, 'duplicate forecast'):
            se.evaluate_sealed_cohort(forecast_seals=[seal(BASE)]*20, label_receipt=labels)

    def test_support_is_not_independence_and_rounded_review_budget_is_exact(self):
        times, flux = complete_series([BASE], [BASE]); labels = self.labels(times, flux)
        result = se.evaluate_sealed_cohort(forecast_seals=[seal(BASE)], label_receipt=labels, minimum_positive_support=1)
        self.assertTrue(result['support_gate_passed'])
        self.assertFalse(result['independent_evaluation_verified'])
        self.assertEqual(result['states']['FULL']['review_enrichment_vs_random'], 1)

    def test_rehashed_invalid_probability_rejected(self):
        original = seal(BASE); original['probabilities']['FULL'] = 2
        unsigned = dict(original); unsigned.pop('forecast_seal_sha256')
        original['forecast_seal_sha256'] = se.sha256_bytes(se._canonical_json(unsigned))
        with self.assertRaisesRegex(se.SealedEvaluationError, 'outside'):
            se.validate_forecast_seal(original)

    def test_target_cannot_change_under_same_name(self):
        times, flux = complete_series([BASE], [])
        with self.assertRaisesRegex(se.SealedEvaluationError, 'frozen target'):
            se.derive_new_crossing_labels(forecast_seals=[seal(BASE)], proton_times=times, proton_flux=flux, threshold_pfu=20)

    def test_same_issue_with_different_prediction_cannot_count_twice(self):
        times, flux = complete_series([BASE], [BASE])
        with self.assertRaisesRegex(se.SealedEvaluationError, 'duplicate forecast issue'):
            se.derive_new_crossing_labels(forecast_seals=[seal(BASE), seal(BASE, full=0.2)], proton_times=times, proton_flux=flux)

    def test_model_package_cannot_change_between_eligible_issues(self):
        first = seal(BASE)
        second = seal(BASE + timedelta(days=1))
        second['package_manifest_sha256'] = 'e' * 64
        unsigned = dict(second); unsigned.pop('forecast_seal_sha256')
        second['forecast_seal_sha256'] = se.sha256_bytes(se._canonical_json(unsigned))
        times, flux = complete_series([BASE, BASE + timedelta(days=1)], [BASE])
        labels = se.derive_new_crossing_labels(forecast_seals=[first, second], proton_times=times, proton_flux=flux)
        with self.assertRaisesRegex(se.SealedEvaluationError, 'model package changed'):
            se.evaluate_sealed_cohort(forecast_seals=[first, second], label_receipt=labels)
