import unittest
from datetime import datetime, timezone

from iris_report.iris_sep.src.iris_sep.pilot_admission_v2 import AdmissionPolicyV2, replay_forecast_v2
from iris_report.iris_sep.tests.test_pilot_replay import fixture


def request():
    value = fixture()
    for record in value["data_freshness"].values():
        record["source_revision"] = "fixture-v1"
    return value


def policy():
    return AdmissionPolicyV2(datetime(2025, 1, 1, tzinfo=timezone.utc),
                             datetime(2027, 1, 1, tzinfo=timezone.utc),
                             {name: ("fixture-v1",) for name in
                              ("magnetic", "eruption", "particle_context")}, 20.0)


class AdmissionV2Tests(unittest.TestCase):
    def test_supported_input_is_valid(self):
        result = replay_forecast_v2(admission_policy=policy(),
            transformed_features=[-3.0, 2.0], model_outputs=[0.3], **request())
        self.assertEqual(result["forecast_status"], "VALID")

    def test_era_revision_magnitude_and_nonfinite_fail_closed(self):
        cases = []
        era = request(); era["issued_at"] = datetime(2028, 1, 1, tzinfo=timezone.utc); cases.append(era)
        revision = request(); revision["data_freshness"]["magnetic"]["source_revision"] = "unknown"; cases.append(revision)
        cases.append((request(), [21.0], [0.3]))
        cases.append((request(), [3.0], [float("nan")]))
        for case in cases:
            if isinstance(case, tuple): item, features, outputs = case
            else: item, features, outputs = case, [3.0], [0.3]
            with self.subTest():
                result = replay_forecast_v2(admission_policy=policy(), transformed_features=features,
                                            model_outputs=outputs, **item)
                self.assertEqual(result["forecast_status"], "ABSTAIN")
                self.assertIsNone(result["operator_state"])


if __name__ == "__main__":
    unittest.main()
