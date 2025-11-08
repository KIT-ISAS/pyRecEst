import unittest
from math import pi

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, mod
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)


class TestToroidalWrappedNormalDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0])
        self.C = array([[1.3, -0.9], [-0.9, 1.2]])
        self.twn = ToroidalWrappedNormalDistribution(self.mu, self.C)

    def test_sanity_check(self):
        self.assertIsInstance(self.twn, ToroidalWrappedNormalDistribution)
        self.assertTrue(allclose(self.twn.mu, self.mu))
        self.assertTrue(allclose(self.twn.C, self.C))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_integrate(self):
        self.assertAlmostEqual(self.twn.integrate(), 1, delta=1e-5)
        self.assertTrue(
            allclose(self.twn.trigonometric_moment(0), array([1.0, 1.0]), rtol=1e-5)
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sampling(self):
        n_samples = 5
        s = self.twn.sample(n_samples)
        self.assertEqual(s.shape, (n_samples, 2))
        self.assertTrue(allclose(s, mod(s, 2 * pi)))


if __name__ == "__main__":
    unittest.main()
