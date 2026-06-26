import json
import unittest

import numpy as np
from pyrecest.evaluation.diagnostic_summaries import top_residuals


class DiagnosticSummaryComplexJsonTest(unittest.TestCase):
    def test_top_residual_records_normalize_complex_payloads(self):
        records = [
            {
                "time_s": 0.0,
                "track_id": "A",
                "source": "rf",
                "residual_norm": 2.0,
                "covariance_scale": 1.0,
                "error": 1.0,
                "phasor": 1.0 + 2.0j,
                "complex_vector": np.array([3.0 + 4.0j, np.nan + 1.0j]),
            }
        ]

        rows = top_residuals(records)

        self.assertEqual(rows[0]["phasor"], {"real": 1.0, "imag": 2.0})
        self.assertEqual(
            rows[0]["complex_vector"],
            [{"real": 3.0, "imag": 4.0}, None],
        )
        json.dumps(rows)


if __name__ == "__main__":
    unittest.main()
