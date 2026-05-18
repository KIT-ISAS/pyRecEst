import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, to_numpy
from pyrecest.distributions import BinghamDistribution

from .test_von_mises_fisher_distribution import vectors_to_test_2d


class TestBinghamDistribution(unittest.TestCase):
    def setUp(self):
        """Setup BinghamDistribution instance for testing."""
        M = array(
            [[1 / 3, 2 / 3, -2 / 3], [-2 / 3, 2 / 3, 1 / 3], [2 / 3, 1 / 3, 2 / 3]]
        )
        Z = array([-5.0, -3.0, 0.0])
        self.bd = BinghamDistribution(Z, M)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_pdf(self):
        """Test pdf method with a fixed set of values."""
        expected_values = array(
            [
                0.0767812166360095,
                0.0145020985787277,
                0.0394207910410773,
                0.0267197897401937,
                0.0298598745474396,
                0.0298598745474396,
            ],
        )
        computed_values = self.bd.pdf(vectors_to_test_2d)
        npt.assert_array_almost_equal(
            computed_values,
            expected_values,
            err_msg="Expected and computed pdf values do not match.",
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_spherical_moment_axis_and_sigma_points(self):
        """S2 Bingham derivatives should support moment-based methods."""
        expected_diagonal = array(
            [
                0.11516984173035563,
                0.1952701622411956,
                0.6895599960284488,
            ]
        )

        moment = self.bd.moment()
        moment_principal_frame = self.bd.M.T @ moment @ self.bd.M

        npt.assert_allclose(
            to_numpy(moment_principal_frame),
            to_numpy(diag(expected_diagonal)),
            rtol=1e-6,
            atol=1e-7,
        )

        axis = self.bd.mean_axis()
        expected_axis = self.bd.M[:, -1]
        axis_alignment = float(to_numpy(axis @ expected_axis))
        self.assertAlmostEqual(abs(axis_alignment), 1.0, places=6)

        samples, weights = self.bd.sample_deterministic()
        self.assertEqual(samples.shape, (3, 6))
        npt.assert_allclose(to_numpy(weights).sum(), 1.0, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
