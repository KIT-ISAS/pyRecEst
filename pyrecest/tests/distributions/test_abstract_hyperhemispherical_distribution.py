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
    def setUp(self):
        self.mu_ = np.array([0.5, 1.0, 1.0]) / np.linalg.norm([0.5, 1.0, 1.0])
        self.kappa_ = 2.0

    def test_get_manifold_size(self):
        """Tests get_manifold_size function with different dimensions."""
        dimensions = [(1, np.pi), (2, 2 * np.pi)]
        for dim, expected in dimensions:
            with self.subTest(dim=dim):
                hud = HyperhemisphericalUniformDistribution(dim)
                self.assertAlmostEqual(hud.get_manifold_size(), expected, delta=1e-16)

    def test_mode_numerical(self):
        """Tests mode_numerical."""
        watson_dist = HyperhemisphericalWatsonDistribution(self.mu_, self.kappa_)
        mode_numerical = watson_dist.mode_numerical()
        np.testing.assert_array_almost_equal(self.mu_, mode_numerical, decimal=6)

    def test_sample_metropolis_hastings_basics_only(self):
        """Tests the sample_metropolis_hastings sampling"""
        vmf = VonMisesFisherDistribution(np.array([1, 0, 0]), 2)
        chd = CustomHyperhemisphericalDistribution(
            lambda x: vmf.pdf(x) + vmf.pdf(-x), vmf.dim
        )
        n = 10
        samples = [chd.sample_metropolis_hastings(n), chd.sample(n)]
        for s in samples:
            with self.subTest(sample=s):
                self.assertEqual(s.shape, (n, chd.input_dim))
                np.testing.assert_allclose(
                    np.sum(s**2, axis=1), np.ones(n), rtol=1e-10
                )


if __name__ == "__main__":
    unittest.main()
