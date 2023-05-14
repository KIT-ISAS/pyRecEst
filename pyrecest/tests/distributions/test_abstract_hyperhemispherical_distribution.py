import unittest

import numpy as np
from pyrecest.distributions import (
    HyperhemisphericalWatsonDistribution,
    VonMisesFisherDistribution,
)
from pyrecest.distributions.hypersphere_subset.custom_hyperhemispherical_distribution import (
    CustomHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)


class TestAbstractHyperhemisphericalDistribution(unittest.TestCase):
    def test_get_manifold_size(self):
        # Circle
        hud = HyperhemisphericalUniformDistribution(1)
        self.assertAlmostEqual(hud.get_manifold_size(), np.pi, delta=1e-16)
        # Sphere
        hud = HyperhemisphericalUniformDistribution(2)
        self.assertAlmostEqual(hud.get_manifold_size(), 2 * np.pi, delta=1e-16)

    def test_mode_numerical(self):
        mu_ = np.array([0.5, 1.0, 1.0] / np.linalg.norm([0.5, 1.0, 1.0]))
        kappa_ = 2.0
        watson_dist = HyperhemisphericalWatsonDistribution(mu_, kappa_)

        mode_numerical = watson_dist.mode_numerical()

        np.testing.assert_array_almost_equal(mu_, mode_numerical, decimal=6)

    def test_sample_metropolis_hastings_basics_only(self):
        vmf = VonMisesFisherDistribution(np.array([1, 0, 0]), 2)
        chd = CustomHyperhemisphericalDistribution(
            lambda x: vmf.pdf(x) + vmf.pdf(-x), vmf.dim
        )
        n = 10
        s = chd.sample_metropolis_hastings(n)
        self.assertEqual(s.shape, (n, chd.input_dim))
        np.testing.assert_allclose(np.sum(s**2, axis=1), np.ones(n), rtol=1e-10)

        s2 = chd.sample(n)
        self.assertEqual(s2.shape, (n, chd.input_dim))
        np.testing.assert_allclose(np.sum(s**2, axis=1), np.ones(n), rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
