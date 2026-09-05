import hashlib
import json
import unittest
from datetime import datetime, timezone

from iris_report.iris_sep.src.iris_sep.inference_bundle import InferenceBundleError, build_inference_bundle, replay_inference_bundle, static_inference_binding_sha256
from iris_report.iris_sep.src.iris_sep.pilot_admission_v2 import AdmissionPolicyV2
from iris_report.iris_sep.workstreams.luna_i_eval_ops.operator import OperatorRuntimePolicy


def fixture():
    runtime = OperatorRuntimePolicy('fixture-policy','fixture-calibration','b'*64,
        {'MONITOR':.2,'PREPARE':.5,'PROTECT':.75},{'magnetic':120,'eruption':60,'particle_context':15},('magnetic',))
    admission = AdmissionPolicyV2(datetime(2025,1,1,tzinfo=timezone.utc),datetime(2027,1,1,tzinfo=timezone.utc),
        {m:('fixture-v1',) for m in runtime.maximum_age_minutes},20.0)
    binding=static_inference_binding_sha256(admission_policy=admission,runtime_policy=runtime,calibration_intercept=0.05,model_version='fixture-model',input_schema_sha256=runtime.schema_sha256)
    receipt={'scope':'SYNTHETIC_FIXTURE_ONLY','locked_test_accessed':False,'inference_binding_sha256':binding,'bindings':{
        'model_version':'fixture-model','schema_sha256':runtime.schema_sha256,'policy_id':runtime.policy_id,
        'calibration_id':runtime.calibration_id,'operating_thresholds':dict(runtime.operating_thresholds),
        'maximum_age_minutes':dict(runtime.maximum_age_minutes),'critical_modalities':list(runtime.critical_modalities)}}
    evidence=json.dumps(receipt,sort_keys=True).encode();fresh={m:{'observed_at_utc':'2026-01-01T11:55:00Z','published_at_utc':'2026-01-01T11:58:00Z','source_revision':'fixture-v1'} for m in runtime.maximum_age_minutes}
    args=dict(admission_policy=admission,runtime_policy=runtime,source_revisions={m:'fixture-v1' for m in runtime.maximum_age_minutes},
        transformed_features=[-3.0,2.0],model_outputs=[-0.2,0.1,0.4],calibration_intercept=0.05,evidence_bytes=evidence,
        expected_evidence_sha256=hashlib.sha256(evidence).hexdigest(),issued_at=datetime(2026,1,1,12,tzinfo=timezone.utc),
        input_schema_sha256=runtime.schema_sha256,data_freshness=fresh,missing_modalities=[],
        uncertainty={'between_seed_spread':.03,'calibration_uncertainty':.04,'input_quality':1.0},model_version='fixture-model')
    return args


class BundleTests(unittest.TestCase):
    def test_valid_bundle_replays_without_external_policy_or_arrays(self):
        bundle,digest=build_inference_bundle(**fixture());result=replay_inference_bundle(bundle_bytes=bundle,expected_bundle_sha256=digest)
        self.assertEqual(result['forecast_status'],'VALID');self.assertEqual(result['inference_bundle_sha256'],digest)
    def test_outer_tamper_is_rejected(self):
        bundle,digest=build_inference_bundle(**fixture())
        with self.assertRaises(InferenceBundleError): replay_inference_bundle(bundle_bytes=bundle+b' ',expected_bundle_sha256=digest)
    def test_inner_array_tamper_is_rejected_even_with_new_outer_anchor(self):
        bundle,_=build_inference_bundle(**fixture());env=json.loads(bundle);env['payload']['arrays']['model_outputs']['bytes_b64']='AAAAAAAAAAA='
        env['payload_sha256']=hashlib.sha256(json.dumps(env['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();mutated=json.dumps(env,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        with self.assertRaises(InferenceBundleError): replay_inference_bundle(bundle_bytes=mutated,expected_bundle_sha256=hashlib.sha256(mutated).hexdigest())
    def test_revision_snapshot_cannot_disagree_with_freshness(self):
        bundle,_=build_inference_bundle(**fixture());env=json.loads(bundle);env['payload']['source_revisions']['magnetic']='other'
        env['payload_sha256']=hashlib.sha256(json.dumps(env['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();mutated=json.dumps(env,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        with self.assertRaises(InferenceBundleError): replay_inference_bundle(bundle_bytes=mutated,expected_bundle_sha256=hashlib.sha256(mutated).hexdigest())
    def test_policy_or_threshold_mutation_breaks_embedded_receipt_binding(self):
        bundle,_=build_inference_bundle(**fixture());env=json.loads(bundle);env['payload']['runtime_policy']['operating_thresholds']['MONITOR']=.25;env['payload']['threshold']['operating_thresholds']['MONITOR']=.25
        env['payload_sha256']=hashlib.sha256(json.dumps(env['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();mutated=json.dumps(env,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        with self.assertRaises(InferenceBundleError): replay_inference_bundle(bundle_bytes=mutated,expected_bundle_sha256=hashlib.sha256(mutated).hexdigest())
    def test_calibration_mutation_is_rejected_by_receipt_static_binding(self):
        import math
        bundle,_=build_inference_bundle(**fixture());env=json.loads(bundle);env['payload']['calibration']['intercept']=0.15
        aggregated=0.1;env['payload']['derived']['calibrated_probability']=1/(1+math.exp(-(aggregated+0.15)))
        env['payload_sha256']=hashlib.sha256(json.dumps(env['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();mutated=json.dumps(env,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        with self.assertRaises(InferenceBundleError): replay_inference_bundle(bundle_bytes=mutated,expected_bundle_sha256=hashlib.sha256(mutated).hexdigest())
    def test_builder_rejects_unbound_evidence_receipt(self):
        args=fixture();receipt=json.loads(args['evidence_bytes']);receipt.pop('inference_binding_sha256');evidence=json.dumps(receipt,sort_keys=True).encode();args['evidence_bytes']=evidence;args['expected_evidence_sha256']=hashlib.sha256(evidence).hexdigest()
        with self.assertRaises(InferenceBundleError): build_inference_bundle(**args)
    def test_builder_rejects_nonfinite_arrays(self):
        args=fixture();args['model_outputs']=[float('nan')]
        with self.assertRaises(InferenceBundleError): build_inference_bundle(**args)

if __name__=='__main__': unittest.main()
