import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import (
    SE2PartiallyWrappedNormalDistribution as ExportedSE2PartiallyWrappedNormalDistribution,
)
from pyrecest.distributions.cart_prod.se2_pwn_distribution import SE2PWNDistribution
from pyrecest.distributions.se2_partially_wrapped_normal_distribution import (
    SE2PartiallyWrappedNormalDistribution,
)


class TestSE2PartiallyWrappedNormalDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0, 3.0])
        self.C = array([[0.9, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
        self.dist = SE2PartiallyWrappedNormalDistribution(self.mu, self.C)
        self.legacy_dist = SE2PWNDistribution(self.mu, self.C)

    def test_top_level_export(self):
        self.assertIs(
            ExportedSE2PartiallyWrappedNormalDistribution,
            SE2PartiallyWrappedNormalDistribution,
        )

    def test_is_legacy_compatible_subclass(self):
        self.assertTrue(
            issubclass(SE2PartiallyWrappedNormalDistribution, SE2PWNDistribution)
        )
        self.assertIsInstance(self.dist, SE2PWNDistribution)

    def test_pdf_matches_legacy_distribution(self):
        points = array([[1.0, 2.0, 3.0], [1.1, 1.9, 2.8], [0.3, -0.2, 4.1]])
        npt.assert_allclose(
            np.asarray(self.dist.pdf(points)), np.asarray(self.legacy_dist.pdf(points))
        )

    def test_mean_4d_matches_legacy_api(self):
        npt.assert_allclose(
            np.asarray(self.dist.mean_4d()), np.asarray(self.legacy_dist.mean4D())
        )

    def test_covariance_4d_matches_legacy_api(self):
        npt.assert_allclose(
            np.asarray(self.dist.covariance_4d()),
            np.asarray(self.legacy_dist.covariance4D()),
        )

    def test_covariance_4d_numerical_matches_legacy_api(self):
        np.random.seed(0)
        cov_new = np.asarray(self.dist.covariance_4d_numerical(20000))
        np.random.seed(0)
        cov_legacy = np.asarray(self.legacy_dist.covariance4D_numerical(20000))
        npt.assert_allclose(cov_new, cov_legacy, atol=5e-2)

    def test_from_samples_returns_descriptive_class(self):
        np.random.seed(42)
        samples = np.asarray(self.legacy_dist.sample(50000))
        fitted = SE2PartiallyWrappedNormalDistribution.from_samples(samples)
        self.assertIsInstance(fitted, SE2PartiallyWrappedNormalDistribution)
        npt.assert_allclose(np.asarray(fitted.mu), np.asarray(self.mu), atol=0.05)
        npt.assert_allclose(np.asarray(fitted.C), np.asarray(self.C), atol=0.1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported for JAX backend",
    )
    def test_integrate(self):
        self.assertAlmostEqual(self.dist.integrate(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
