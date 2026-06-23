import unittest

import numpy as np

from pyrecest.models.validation import validate_covariance_matrix


class CovarianceToleranceTest(unittest.TestCase):
    def test_symmetry_tolerances_reject_invalid_scalars(self):
        covariance = np.eye(2)
        invalid_values = (-1.0, np.nan, np.inf, np.array([1.0]))

        for value in invalid_values:
            with self.subTest(parameter="symmetric_atol", value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "symmetric_atol must be a finite nonnegative scalar",
                ):
                    validate_covariance_matrix(
                        covariance,
                        check_symmetric=True,
                        symmetric_atol=value,
                    )
            with self.subTest(parameter="symmetric_rtol", value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "symmetric_rtol must be a finite nonnegative scalar",
                ):
                    validate_covariance_matrix(
                        covariance,
                        check_symmetric=True,
                        symmetric_rtol=value,
                    )


if __name__ == "__main__":
    unittest.main()
