import unittest

import numpy as np
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.custom_hyperhemispherical_distribution import (
    CustomHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_mixture import (
    HypersphericalMixture,
)
from pyrecest.distributions.hypertorus.hypertoroidal_mixture import HypertoroidalMixture
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)


class AbstractMixtureTest(unittest.TestCase):
    def _test_sample(self, mix, n):
        for sampling_method in [mix.sample_metropolis_hastings, mix.sample]:
            s = sampling_method(n)
            self.assertEqual(s.shape, (n, mix.input_dim))
        return s

    def test_sample_metropolis_hastings_basics_only_t2(self):
        vmf = ToroidalWrappedNormalDistribution(np.array([1, 0]), np.eye(2))
        mix = HypertoroidalMixture([vmf, vmf.shift(np.array([1, 1]))],np.array([0.5, 0.5]))
        self._test_sample(mix, 10)

    def test_sample_metropolis_hastings_basics_only_s2(self):
        vmf1 = VonMisesFisherDistribution(np.array([1, 0, 0]), 2)
        vmf2 = VonMisesFisherDistribution(np.array([0, 1, 0]), 2)
        mix = HypersphericalMixture([vmf1, vmf2], [0.5, 0.5])
        s = self._test_sample(mix, 10)
        self.assertTrue(np.allclose(np.linalg.norm(s, axis=1), np.ones(10), rtol=1e-10))

    def test_sample_metropolis_hastings_basics_only_h2(self):
        vmf = VonMisesFisherDistribution(np.array([1, 0, 0]), 2)
        mix = CustomHyperhemisphericalDistribution(
            lambda x: vmf.pdf(x) + vmf.pdf(-x), 2
        )
        s = self._test_sample(mix, 10)
        self.assertTrue(np.allclose(np.linalg.norm(s, axis=1), np.ones(10), rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
