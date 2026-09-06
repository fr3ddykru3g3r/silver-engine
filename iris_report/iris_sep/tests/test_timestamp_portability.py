import unittest

import numpy as np
import pandas as pd

from iris_report.iris_sep.tools import run_public_new_crossing_benchmark as benchmark


class TimestampPortabilityTests(unittest.TestCase):
    def test_unix_seconds_identical_for_ns_and_us_storage(self):
        base = pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="6h")
        ns = pd.Series(base.as_unit("ns"))
        us = pd.Series(base.as_unit("us"))
        self.assertEqual(str(ns.dtype), "datetime64[ns, UTC]")
        self.assertEqual(str(us.dtype), "datetime64[us, UTC]")
        np.testing.assert_array_equal(
            benchmark._unix_seconds(ns),
            benchmark._unix_seconds(us),
        )

    def test_new_crossing_target_identical_for_ns_and_us_storage(self):
        issue_times = pd.DatetimeIndex(
            pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T12:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-02T12:00:00Z",
                    "2026-01-03T00:00:00Z",
                ],
                utc=True,
            )
        )
        events = pd.DataFrame(
            {
                benchmark.EVENT_START: ["2026/01/02 00:00", "2026/01/04 00:00"],
                benchmark.EVENT_END: ["2026/01/02 06:00", "2026/01/04 03:00"],
            }
        )

        outputs = []
        for unit in ("ns", "us"):
            frame = pd.DataFrame({"window_end": issue_times.as_unit(unit)})
            target, active, event_ids, parsed = benchmark.derive_target(frame, events)
            outputs.append((target, active, event_ids, parsed))

        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
        np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
        pd.testing.assert_frame_equal(outputs[0][3], outputs[1][3])

        # Exactly +24 h counts as a positive NEW crossing.
        self.assertEqual(int(outputs[0][0][0]), 1)
        # An issue exactly at event onset is ineligible because the event is active.
        self.assertTrue(bool(outputs[0][1][2]))
        self.assertEqual(int(outputs[0][0][2]), 0)

    def test_event_identifier_preserves_historical_epoch_second_definition(self):
        issue = pd.Timestamp("2026-01-01T00:00:00Z")
        onset = pd.Timestamp("2026-01-02T00:00:00Z")
        events = pd.DataFrame(
            {
                benchmark.EVENT_START: ["2026/01/02 00:00"],
                benchmark.EVENT_END: ["2026/01/02 01:00"],
            }
        )
        frame = pd.DataFrame({"window_end": pd.DatetimeIndex([issue]).as_unit("us")})
        target, active, event_ids, _ = benchmark.derive_target(frame, events)
        expected_seconds = int(onset.value // 1_000_000_000)
        import hashlib
        expected_id = hashlib.sha256(str(expected_seconds).encode()).hexdigest()[:24]
        self.assertEqual(target.tolist(), [1])
        self.assertEqual(active.tolist(), [False])
        self.assertEqual(event_ids.tolist(), [expected_id])


if __name__ == "__main__":
    unittest.main()
