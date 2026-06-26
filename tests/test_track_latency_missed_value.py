"""Regression tests for track-latency missed-value validation."""

import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.utils.track_metrics import track_latencies


class TestTrackLatencyMissedValue(unittest.TestCase):
    def test_track_latency_rejects_invalid_missed_values(self):
        invalid_missed_values = (
            True,
            "1.0",
            b"1.0",
            np.inf,
            -np.inf,
            np.array([np.nan]),
        )

        for missed_value in invalid_missed_values:
            with self.subTest(missed_value=missed_value):
                with self.assertRaisesRegex(ValueError, "missed_value"):
                    track_latencies([[None]], [[0]], missed_value=missed_value)

    def test_track_latency_accepts_numeric_missed_values(self):
        npt.assert_allclose(
            track_latencies([[None]], [[0]], missed_value=7.0), np.array([7.0])
        )
        values = track_latencies([[None]], [[0]], missed_value=np.array(np.nan))

        self.assertEqual(values.shape, (1,))
        self.assertTrue(np.isnan(values[0]))


if __name__ == "__main__":
    unittest.main()
