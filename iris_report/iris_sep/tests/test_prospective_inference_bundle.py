from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime, timezone

from iris_report.iris_sep.src.iris_sep.inference_bundle import static_inference_binding_sha256
from iris_report.iris_sep.src.iris_sep.pilot_admission_v2 import AdmissionPolicyV2
from iris_report.iris_sep.src.iris_sep.prospective_inference_bundle import (
    ProspectiveInferenceBundleError,
    build_prospective_inference_bundle,
    replay_prospective_inference_bundle,
)
from iris_report.iris_sep.workstreams.luna_i_eval_ops.operator import OperatorRuntimePolicy


ISSUED = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def inference_fixture():
    runtime = OperatorRuntimePolicy(
        "fixture-policy",
        "fixture-calibration",
        "b" * 64,
        {"MONITOR": 0.2, "PREPARE": 0.5, "PROTECT": 0.75},
        {"magnetic": 120, "eruption": 60, "particle_context": 15},
        ("magnetic",),
    )
    admission = AdmissionPolicyV2(
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2027, 1, 1, tzinfo=timezone.utc),
        {m: ("fixture-v1",) for m in runtime.maximum_age_minutes},
        20.0,
    )
    binding = static_inference_binding_sha256(
        admission_policy=admission,
        runtime_policy=runtime,
        calibration_intercept=0.05,
        model_version="fixture-model",
        input_schema_sha256=runtime.schema_sha256,
    )
    evidence = json.dumps(
        {
            "scope": "SYNTHETIC_FIXTURE_ONLY",
            "locked_test_accessed": False,
            "inference_binding_sha256": binding,
        },
        sort_keys=True,
    ).encode()
    freshness = {
        modality: {
            "observed_at_utc": "2026-01-01T11:55:00Z",
            "published_at_utc": "2026-01-01T11:58:00Z",
            "source_revision": "fixture-v1",
        }
        for modality in runtime.maximum_age_minutes
    }
    return {
        "admission_policy": admission,
        "runtime_policy": runtime,
        "source_revisions": {m: "fixture-v1" for m in runtime.maximum_age_minutes},
        "transformed_features": [-3.0, 2.0],
        "model_outputs": [-0.2, 0.1, 0.4],
        "calibration_intercept": 0.05,
        "evidence_bytes": evidence,
        "expected_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
        "issued_at": ISSUED,
        "input_schema_sha256": runtime.schema_sha256,
        "data_freshness": freshness,
        "missing_modalities": [],
        "uncertainty": {
            "between_seed_spread": 0.03,
            "calibration_uncertainty": 0.04,
            "input_quality": 1.0,
        },
        "model_version": "fixture-model",
    }


def provenance_records(state="NATIVE_OBSERVED"):
    return [
        {
            "feature_name": "xrs_feature",
            "family": "XRS",
            "state": state,
            "source_revision": "goes-xrs-fixture",
            "observed_at_utc": "2026-01-01T11:50:00Z",
            "published_at_utc": "2026-01-01T11:55:00Z",
            "uses_future_values": False,
            "retrospective_fit": False,
            "transform_fit_through_utc": None,
        },
        {
            "feature_name": "proton_feature",
            "family": "PROTON",
            "state": "ALTERNATE_SOURCE_OBSERVED",
            "source_revision": "goes-proton-fixture",
            "observed_at_utc": "2026-01-01T11:50:00Z",
            "published_at_utc": "2026-01-01T11:55:00Z",
            "uses_future_values": False,
            "retrospective_fit": False,
            "transform_fit_through_utc": None,
        },
    ]


REQUIRED = {"XRS": ("xrs_feature",), "PROTON": ("proton_feature",)}


def reanchor(envelope):
    envelope["payload_sha256"] = hashlib.sha256(
        json.dumps(envelope["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    mutated = json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return mutated, hashlib.sha256(mutated).hexdigest()


class ProspectiveInferenceBundleTests(unittest.TestCase):
    def test_valid_provenance_is_bound_and_recomputed_on_replay(self):
        bundle, digest = build_prospective_inference_bundle(
            input_provenance_records=provenance_records(),
            required_features=REQUIRED,
            development_information_cutoff=None,
            **inference_fixture(),
        )
        result = replay_prospective_inference_bundle(
            bundle_bytes=bundle,
            expected_bundle_sha256=digest,
        )
        self.assertEqual(result["forecast_status"], "VALID")
        self.assertTrue(result["forecast_probability_permitted_by_provenance"])
        self.assertEqual(result["forecast_input_provenance"]["status"], "VALID")
        self.assertEqual(result["prospective_inference_bundle_sha256"], digest)

    def test_builder_blocks_unknown_aggregate_lineage(self):
        with self.assertRaises(ProspectiveInferenceBundleError):
            build_prospective_inference_bundle(
                input_provenance_records=provenance_records(state="UNKNOWN"),
                required_features=REQUIRED,
                development_information_cutoff=None,
                **inference_fixture(),
            )

    def test_builder_requires_lineage_for_every_transformed_feature(self):
        with self.assertRaises(ProspectiveInferenceBundleError):
            build_prospective_inference_bundle(
                input_provenance_records=provenance_records(),
                required_features={"XRS": ("xrs_feature",)},
                development_information_cutoff=None,
                **inference_fixture(),
            )

    def test_replay_rejects_tampered_raw_lineage_even_with_new_outer_anchor(self):
        bundle, _ = build_prospective_inference_bundle(
            input_provenance_records=provenance_records(),
            required_features=REQUIRED,
            development_information_cutoff=None,
            **inference_fixture(),
        )
        envelope = json.loads(bundle)
        envelope["payload"]["provenance"]["records"][0]["state"] = "UNKNOWN"
        mutated, anchor = reanchor(envelope)
        with self.assertRaises(ProspectiveInferenceBundleError):
            replay_prospective_inference_bundle(
                bundle_bytes=mutated,
                expected_bundle_sha256=anchor,
            )

    def test_replay_rejects_forged_saved_gate_even_with_new_outer_anchor(self):
        bundle, _ = build_prospective_inference_bundle(
            input_provenance_records=provenance_records(),
            required_features=REQUIRED,
            development_information_cutoff=None,
            **inference_fixture(),
        )
        envelope = json.loads(bundle)
        envelope["payload"]["provenance"]["gate_receipt"]["manifest_sha256"] = "0" * 64
        mutated, anchor = reanchor(envelope)
        with self.assertRaises(ProspectiveInferenceBundleError):
            replay_prospective_inference_bundle(
                bundle_bytes=mutated,
                expected_bundle_sha256=anchor,
            )

    def test_replay_rejects_provenance_issue_time_different_from_inner_request(self):
        bundle, _ = build_prospective_inference_bundle(
            input_provenance_records=provenance_records(),
            required_features=REQUIRED,
            development_information_cutoff=None,
            **inference_fixture(),
        )
        envelope = json.loads(bundle)
        envelope["payload"]["provenance"]["issued_at"] = "2026-01-01T11:59:00+00:00"
        mutated, anchor = reanchor(envelope)
        with self.assertRaises(ProspectiveInferenceBundleError):
            replay_prospective_inference_bundle(
                bundle_bytes=mutated,
                expected_bundle_sha256=anchor,
            )


if __name__ == "__main__":
    unittest.main()
