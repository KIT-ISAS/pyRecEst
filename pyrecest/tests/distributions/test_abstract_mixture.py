import unittest
import numpy as np
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import ToroidalWrappedNormalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_mixture import HypersphericalMixture
from pyrecest.distributions.hypersphere_subset.custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution

class AbstractMixtureTest(unittest.TestCase):

    def test_sample_metropolis_hastings_basics_only_t2(self):
        vmf = ToroidalWrappedNormalDistribution(np.array([1, 0]), np.eye(2))
        mix = HypertoroidalMixture([vmf, vmf.shift(np.array([1, 1]))], [0.5, 0.5])
        n = 10
        s = mix.sample_metropolis_hastings(n)
        self.assertEqual(s.shape, (n, mix.dim))

        s2 = mix.sample(n)
        self.assertEqual(s2.shape, (n, mix.dim))

    def test_sample_metropolis_hastings_basics_only_s2(self):
        vmf1 = VonMisesFisherDistribution(np.array([1, 0, 0]), 2)
        vmf2 = VonMisesFisherDistribution(np.array([0, 1, 0]), 2)
        mix = HypersphericalMixture([vmf1, vmf2], [0.5, 0.5])
        n = 10
        s = mix.sample_metropolis_hastings(n)
        self.assertEqual(s.shape, (n, mix.input_dim))
        self.assertTrue(np.allclose(np.linalg.norm(s, axis=1), np.ones(n), rtol=1e-10))

        s2 = mix.sample(n)
        self.assertEqual(s2.shape, (n, mix.input_dim))
        self.assertTrue(np.allclose(np.linalg.norm(s, axis=1), np.ones(n), rtol=1e-10))

    def test_sample_metropolis_hastings_basics_only_h2(self):
        vmf = VonMisesFisherDistribution(np.array([1, 0, 0]), 2)
        mix = CustomHyperhemisphericalDistribution(lambda x: vmf.pdf(x) + vmf.pdf(-x), 2)
        n = 10
        s = mix.sample_metropolis_hastings(n)
        self.assertEqual(s.shape, (n, mix.input_dim))
        self.assertTrue(np.allclose(np.linalg.norm(s, axis=1), np.ones(n), rtol=1e-10))

        s2 = mix.sample(n)
        self.assertEqual(s2.shape, (n, mix.input_dim))
        self.assertTrue(np.allclose(np.linalg.norm(s, axis=1), np.ones(n), rtol=1e-10))

if __name__ == '__main__':
    unittest.main()
