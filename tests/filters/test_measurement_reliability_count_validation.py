import unittest

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.filters import (
    normalize_active_measurement_mask,
    normalize_measurement_noise_covariances,
    normalize_measurement_weights,
)


def _as_covariance_matrix(value, dim, name):
    matrix = array(value)
    if matrix.ndim == 0:
        matrix = matrix * eye(dim)
    if matrix.ndim == 1:
        if matrix.shape[0] != dim:
            raise ValueError(f"{name} vector must have length {dim}")
        matrix = matrix * eye(dim)
    if matrix.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim})")
    return matrix


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="count validation test uses NumPy backend array shape checks",
)
class TestMeasurementReliabilityCountValidation(unittest.TestCase):
    def test_measurement_count_must_be_nonnegative_integer(self):
        invalid_counts = (True, False, 1.5, "2", array([2]), -1)

        for n_measurements in invalid_counts:
            with self.subTest(n_measurements=n_measurements):
                with self.assertRaisesRegex(ValueError, "n_measurements"):
                    normalize_measurement_weights(None, n_measurements)
                with self.assertRaisesRegex(ValueError, "n_measurements"):
                    normalize_active_measurement_mask(None, n_measurements)
                with self.assertRaisesRegex(ValueError, "n_measurements"):
                    normalize_measurement_noise_covariances(
                        0.5,
                        n_measurements,
                        2,
                        as_covariance_matrix=_as_covariance_matrix,
                    )

    def test_measurement_dim_must_be_positive_integer(self):
        invalid_dims = (True, False, 0, -1, 1.5, "2", array([2]))

        for measurement_dim in invalid_dims:
            with self.subTest(measurement_dim=measurement_dim):
                with self.assertRaisesRegex(ValueError, "measurement_dim"):
                    normalize_measurement_noise_covariances(
                        0.5,
                        1,
                        measurement_dim,
                        as_covariance_matrix=_as_covariance_matrix,
                    )


if __name__ == "__main__":
    unittest.main()
