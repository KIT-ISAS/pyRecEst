import unittest

import numpy as np
from pyrecest.calibration.time_offset import make_offset_grid


class TimeOffsetGridInputValidationTest(unittest.TestCase):
    def test_rejects_nonfinite_and_nonscalar_grid_inputs(self):
        invalid_cases = (
            (float("nan"), 1.0, 0.1),
            (0.0, float("inf"), 0.1),
            (0.0, 1.0, float("inf")),
            ([0.0], 1.0, 0.1),
            (0.0, True, 0.1),
            ("0.0", 1.0, 0.1),
            (0.0, np.array(True, dtype=object), 0.1),
        )
        for min_s, max_s, step_s in invalid_cases:
            with self.subTest(min_s=min_s, max_s=max_s, step_s=step_s):
                with self.assertRaisesRegex(ValueError, "must be a finite scalar"):
                    make_offset_grid(min_s, max_s, step_s)

    def test_preserves_step_and_ordering_errors(self):
        with self.assertRaisesRegex(ValueError, "step_s must be positive"):
            make_offset_grid(0.0, 1.0, 0.0)
        with self.assertRaisesRegex(
            ValueError, "max_s must be greater than or equal to min_s"
        ):
            make_offset_grid(1.0, 0.0, 0.1)
