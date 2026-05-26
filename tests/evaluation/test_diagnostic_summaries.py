import json
import math
import unittest

import numpy as np

from pyrecest.evaluation.diagnostic_summaries import (
    build_diagnostic_summary,
    covariance_inflation_summary,
    top_residuals,
    track_switch_summary,
    worst_time_windows,
)


class DiagnosticSummariesTest(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "time_s": 0.0,
                "track_id": "A",
                "source": "rf",
                "residual_norm": 1.0,
                "covariance_scale": 1.0,
                "error": 1.0,
            },
            {
                "time_s": 1.0,
                "track_id": "A",
                "source": "rf",
                "residual_norm": 4.0,
                "covariance_scale": 2.0,
                "error": 2.0,
            },
            {
                "time_s": 2.0,
                "track_id": "B",
                "source": "radar",
                "residual_norm": 3.0,
                "covariance_scale": 3.0,
                "error": 10.0,
            },
            {
                "time_s": 7.0,
                "track_id": "B",
                "source": "radar",
                "residual_norm": 2.0,
                "covariance_scale": 1.0,
                "error": 3.0,
            },
        ]

    def test_top_residuals_returns_largest_first(self):
        rows = top_residuals(self.records, top_n=2)

        self.assertEqual([row["residual_norm"] for row in rows], [4.0, 3.0])

    def test_track_switch_summary_counts_transitions(self):
        summary = track_switch_summary(self.records)

        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["top_transitions"][0]["from_track_id"], "A")
        self.assertEqual(summary["top_transitions"][0]["to_track_id"], "B")

    def test_covariance_inflation_summary_counts_by_source(self):
        summary = covariance_inflation_summary(self.records)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["by_source"], {"radar": 1, "rf": 1})
        self.assertEqual(summary["max_scale"], 3.0)

    def test_worst_windows_sort_by_rmse(self):
        rows = worst_time_windows(self.records, window_s=5.0, top_n=2)

        self.assertEqual(rows[0]["time_start_s"], 0.0)
        self.assertTrue(math.isclose(rows[0]["rmse"], math.sqrt((1 + 4 + 100) / 3)))
        self.assertEqual(rows[0]["track_switch_count"], 1)

    def test_build_diagnostic_summary(self):
        summary = build_diagnostic_summary(self.records, top_n=2, window_s=5.0)

        self.assertEqual(summary["schema_version"], 1)
        self.assertEqual(len(summary["top_residuals"]), 2)
        self.assertEqual(summary["covariance_inflation"]["count"], 2)

    def test_summary_records_are_recursively_json_serializable(self):
        records = [
            {
                "time_s": 0.0,
                "track_id": "A",
                "residual_norm": 1.0,
                "covariance_scale": 2.0,
                "error": 1.0,
                "nested": {
                    "np_int": np.int64(3),
                    "np_bool": np.bool_(True),
                    "np_nan": np.float64(np.nan),
                    "items": (np.float32(1.5), np.array([np.int64(2), np.inf])),
                },
            }
        ]

        summary = build_diagnostic_summary(records, top_n=1)

        json.dumps(summary, allow_nan=False)
        nested = summary["top_residuals"][0]["nested"]
        self.assertEqual(nested["np_int"], 3)
        self.assertIs(nested["np_bool"], True)
        self.assertIsNone(nested["np_nan"])
        self.assertEqual(nested["items"], [1.5, [2, None]])


if __name__ == "__main__":
    unittest.main()
