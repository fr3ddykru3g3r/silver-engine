"""End-to-end synthetic receipt/availability failure injections."""
import copy
import hashlib
import json
import unittest
from datetime import datetime, timezone
from iris_report.iris_sep.src.iris_sep.pilot_replay import replay_forecast
from iris_report.iris_sep.workstreams.luna_i_eval_ops.operator import OperatorRuntimePolicy


def fixture():
    policy = OperatorRuntimePolicy('fixture-policy', 'fixture-calibration', 'b'*64,
        {'MONITOR': .2, 'PREPARE': .5, 'PROTECT': .75},
        {'magnetic': 120, 'eruption': 60, 'particle_context': 15}, ('magnetic',))
    receipt = {'scope': 'SYNTHETIC_FIXTURE_ONLY', 'locked_test_accessed': False,
        'bindings': {'model_version': 'fixture-model', 'schema_sha256': policy.schema_sha256,
            'policy_id': policy.policy_id, 'calibration_id': policy.calibration_id,
            'operating_thresholds': dict(policy.operating_thresholds),
            'maximum_age_minutes': dict(policy.maximum_age_minutes),
            'critical_modalities': list(policy.critical_modalities)}}
    evidence = json.dumps(receipt, sort_keys=True).encode()
    return dict(evidence_bytes=evidence, expected_evidence_sha256=hashlib.sha256(evidence).hexdigest(),
        issued_at=datetime(2026, 1, 1, 12, tzinfo=timezone.utc), calibrated_probability=.3,
        runtime_policy=policy, input_schema_sha256=policy.schema_sha256,
        data_freshness={m: {'observed_at_utc': '2026-01-01T11:55:00Z',
                           'published_at_utc': '2026-01-01T11:58:00Z'} for m in policy.maximum_age_minutes},
        missing_modalities=[], uncertainty={'between_seed_spread': .03, 'calibration_uncertainty': .04,
        'input_quality': 1.0}, model_version='fixture-model')


def cases():
    base = fixture()
    result = {'available': (base, 'VALID')}
    def add(name, **changes):
        value = copy.deepcopy(base); value.update(changes); result[name] = (value, 'ABSTAIN')
    add('missing_evidence', evidence_bytes=None)
    add('mutated_evidence', evidence_bytes=base['evidence_bytes'] + b' ')
    add('schema_mismatch', input_schema_sha256='c'*64)
    add('missing_critical_input', missing_modalities=['magnetic'])
    add('missing_feed_record', data_freshness={})
    add('missing_uncertainty', uncertainty={})
    add('model_mismatch', model_version='other')
    for name, field, timestamp in [('stale_feed', 'observed_at_utc', '2026-01-01T08:00:00Z'),
                                    ('future_publication', 'published_at_utc', '2026-01-01T12:01:00Z')]:
        fresh = copy.deepcopy(base['data_freshness']); fresh['magnetic'][field] = timestamp
        add(name, data_freshness=fresh)
    degraded = copy.deepcopy(base); degraded['missing_modalities'] = ['particle_context']
    result['optional_missing'] = (degraded, 'DEGRADED')
    return result


class ReplayTests(unittest.TestCase):
    def test_failure_replays(self):
        for name, (request, expected) in cases().items():
            with self.subTest(name=name):
                actual = replay_forecast(**copy.deepcopy(request))
                self.assertEqual(actual['forecast_status'], expected)
                json.dumps(actual, allow_nan=False)
                if expected == 'ABSTAIN':
                    self.assertIsNone(actual['operator_state'])
                    self.assertIsNone(actual['p_new_sep_10mev_10pfu_within_24h'])
                    self.assertTrue(actual['abstention_reasons'])
    def test_fixture_receipt_cannot_be_promoted(self):
        request = fixture(); evidence = json.loads(request['evidence_bytes'])
        evidence['scope'] = 'FINAL_BENCHMARK'
        request['evidence_bytes'] = json.dumps(evidence).encode()
        request['expected_evidence_sha256'] = hashlib.sha256(request['evidence_bytes']).hexdigest()
        self.assertEqual(replay_forecast(**request)['forecast_status'], 'ABSTAIN')

if __name__ == '__main__': unittest.main()
