import unittest
from math import pi

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg, mod, ones, random, sqrt, sum
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import (
    HypersphericalDiracDistribution,
)


class HypersphericalDiracDistributionTest(unittest.TestCase):
    def setUp(self):
        self.d = array(
            [
                [0.5, 3.0, 4.0, 6.0, 6.0],
                [2.0, 2.0, 5.0, 3.0, 0.0],
                [0.5, 0.2, 5.8, 4.3, 1.2],
            ]
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
        npt.assert_array_almost_equal(s, mod(s, 2 * pi))
        npt.assert_array_almost_equal(linalg.norm(s, axis=-1), ones(nSamples))

    def test_apply_function(self):
        same = self.hdd.apply_function(lambda x: x)
        npt.assert_array_almost_equal(same.d, self.hdd.d, decimal=10)
        npt.assert_array_almost_equal(same.w, self.hdd.w, decimal=10)

        mirrored = self.hdd.apply_function(lambda x: -x)
        npt.assert_array_almost_equal(mirrored.d, -self.hdd.d, decimal=10)
        npt.assert_array_almost_equal(mirrored.w, self.hdd.w, decimal=10)

    def test_reweigh_identity(self):
        def f(x):
            return 2 * ones(x.shape[0])

        twdNew = self.hdd.reweigh(f)
        self.assertIsInstance(twdNew, HypersphericalDiracDistribution)
        npt.assert_array_almost_equal(twdNew.d, self.hdd.d)
        npt.assert_array_almost_equal(twdNew.w, self.hdd.w)

    def test_reweigh_coord_based(self):
        def f(x):
            return x[:, 1]

        twdNew = self.hdd.reweigh(f)
        self.assertIsInstance(twdNew, HypersphericalDiracDistribution)
        npt.assert_array_almost_equal(twdNew.d, self.hdd.d)
        self.assertAlmostEqual(sum(twdNew.w), 1, places=10)
        wNew = self.hdd.d[:, 1] * self.hdd.w
        npt.assert_array_almost_equal(twdNew.w, wNew / sum(wNew))

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.pytorch",
        reason="Not supported on PyTorch backend",
    )
    def test_from_distribution(self):
        random.seed(0)
        vmf = VonMisesFisherDistribution(array([1.0, 1.0, 1.0]) / sqrt(3), array(1.0))
        dirac_dist = HypersphericalDiracDistribution.from_distribution(vmf, 100000)
        npt.assert_almost_equal(
            dirac_dist.mean_direction(), vmf.mean_direction(), decimal=2
        )


if __name__ == "__main__":
    unittest.main()
