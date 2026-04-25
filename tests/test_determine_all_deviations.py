import unittest
import warnings

import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, isinf
from pyrecest.evaluation import determine_all_deviations


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Not supported on this backend",
)
class TestDetermineAllDeviations(unittest.TestCase):
    def test_missing_estimates_do_not_require_filter_metadata(self):
        groundtruths = np.empty((2, 1), dtype=object)
        groundtruths[0, 0] = array([1.0, 2.0])
        groundtruths[1, 0] = array([3.0, 4.0])
        results = np.empty((1, 2), dtype=object)
        results[0, 0] = None
        results[0, 1] = None

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            deviations = determine_all_deviations(
                results,
                lambda filter_state: filter_state,
                lambda estimate, truth: np.linalg.norm(estimate - truth),
                groundtruths,
            )

        self.assertTrue(np.all(isinf(deviations)))
        self.assertTrue(
            any(
                "Filter result 0 apparently failed 2 times" in str(warning.message)
                for warning in caught_warnings
            )
        )


if __name__ == "__main__":
    unittest.main()
