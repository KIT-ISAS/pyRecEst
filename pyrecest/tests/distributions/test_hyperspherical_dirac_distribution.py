from pyrecest.backend import linalg
from math import pi
from pyrecest.backend import random
from pyrecest.backend import sum
from pyrecest.backend import sqrt
from pyrecest.backend import ones
from pyrecest.backend import mod
from pyrecest.backend import array
import unittest

import numpy as np
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)


class HypersphericalDiracDistributionTest(unittest.TestCase):
    def setUp(self):
        self.d = array(
            [[0.5, 3.0, 4.0, 6.0, 6.0], [2.0, 2.0, 5.0, 3.0, 0.0], [0.5, 0.2, 5.8, 4.3, 1.2]]
        ).T
        self.d = self.d / linalg.norm(self.d, axis=1)[:, None]
        self.w = array([0.1, 0.1, 0.1, 0.1, 0.6])
        self.hdd = HypersphericalDiracDistribution(self.d, self.w)

    def test_instance_creation(self):
        self.assertIsInstance(self.hdd, HypersphericalDiracDistribution)

    def test_sampling(self):
        nSamples = 5
        s = self.hdd.sample(nSamples)
        self.assertEqual(s.shape, (nSamples, self.d.shape[-1]))
        np.testing.assert_array_almost_equal(s, mod(s, 2 * pi))
        np.testing.assert_array_almost_equal(
            linalg.norm(s, axis=-1), ones(nSamples)
        )

    def test_apply_function(self):
        same = self.hdd.apply_function(lambda x: x)
        np.testing.assert_array_almost_equal(same.d, self.hdd.d, decimal=10)
        np.testing.assert_array_almost_equal(same.w, self.hdd.w, decimal=10)

        mirrored = self.hdd.apply_function(lambda x: -x)
        np.testing.assert_array_almost_equal(mirrored.d, -self.hdd.d, decimal=10)
        np.testing.assert_array_almost_equal(mirrored.w, self.hdd.w, decimal=10)

    def test_reweigh_identity(self):
        def f(x):
            return 2 * ones(x.shape[0])

        twdNew = self.hdd.reweigh(f)
        self.assertIsInstance(twdNew, HypersphericalDiracDistribution)
        np.testing.assert_array_almost_equal(twdNew.d, self.hdd.d)
        np.testing.assert_array_almost_equal(twdNew.w, self.hdd.w)

    def test_reweigh_coord_based(self):
        def f(x):
            return x[:, 1]

        twdNew = self.hdd.reweigh(f)
        self.assertIsInstance(twdNew, HypersphericalDiracDistribution)
        np.testing.assert_array_almost_equal(twdNew.d, self.hdd.d)
        self.assertAlmostEqual(sum(twdNew.w), 1, places=10)
        wNew = self.hdd.d[:, 1] * self.hdd.w
        np.testing.assert_array_almost_equal(twdNew.w, wNew / sum(wNew))

    def test_from_distribution(self):
        random.seed(0)
        vmf = VonMisesFisherDistribution(array([1.0, 1.0, 1.0]) / sqrt(3), 1.0)
        dirac_dist = HypersphericalDiracDistribution.from_distribution(vmf, 100000)
        np.testing.assert_almost_equal(
            dirac_dist.mean_direction(), vmf.mean_direction(), decimal=2
        )


if __name__ == "__main__":
    unittest.main()