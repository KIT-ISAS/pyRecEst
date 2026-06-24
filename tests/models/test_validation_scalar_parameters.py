"""Regression tests for model-validation scalar parameters."""

from __future__ import annotations

import unittest

import numpy as np
from pyrecest.models.validation import validate_covariance_matrix


class TestModelValidationScalarParameters(unittest.TestCase):
    def test_symmetry_tolerances_reject_string_like_scalars(self) -> None:
        covariance = [[1.0, 0.0], [0.0, 1.0]]
        invalid_values = (
            "0.0",
            b"0.0",
            np.array("0.0"),
            np.array(b"0.0", dtype="S3"),
            np.array("0.0", dtype=object),
        )

        for tolerance_name in ("symmetric_rtol", "symmetric_atol"):
            for invalid_value in invalid_values:
                with self.subTest(
                    tolerance_name=tolerance_name,
                    invalid_value=repr(invalid_value),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{tolerance_name} must be a finite nonnegative scalar",
                    ):
                        validate_covariance_matrix(
                            covariance,
                            check_symmetric=True,
                            **{tolerance_name: invalid_value},
                        )

    def test_symmetry_tolerances_accept_numeric_scalar_arrays(self) -> None:
        covariance = [[1.0, 0.0], [0.0, 1.0]]

        validated = validate_covariance_matrix(
            covariance,
            check_symmetric=True,
            symmetric_rtol=np.array(0.0),
            symmetric_atol=np.array(0.0),
        )

        self.assertEqual(tuple(validated.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
