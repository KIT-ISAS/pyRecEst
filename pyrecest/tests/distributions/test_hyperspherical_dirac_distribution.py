import unittest
import numpy as np
from pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution import HypersphericalDiracDistribution
from pyrecest.distributions import VonMisesFisherDistribution

class HypersphericalDiracDistributionTest(unittest.TestCase):

    def setUp(self):
        self.d = np.array([[0.5,3,4,6,6],
                           [2,2,5,3,0],
                           [0.5,0.2,5.8,4.3,1.2]]).T
        self.d = self.d / np.linalg.norm(self.d, axis=1)[:, None]
        self.w = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
        self.hdd = HypersphericalDiracDistribution(self.d, self.w)

    def test_instance_creation(self):
        self.assertIsInstance(self.hdd, HypersphericalDiracDistribution)
    
    def test_sampling(self):
        nSamples = 5
        s = self.hdd.sample(nSamples)
        self.assertEqual(s.shape, (nSamples, self.d.shape[-1]))
        np.testing.assert_array_almost_equal(s, np.mod(s, 2*np.pi))
        np.testing.assert_array_almost_equal(np.linalg.norm(s, axis=-1), np.ones(nSamples))
    
    def test_apply_function(self):
        same = self.hdd.apply_function(lambda x: x)
        np.testing.assert_array_almost_equal(same.d, self.hdd.d, decimal=10)
        np.testing.assert_array_almost_equal(same.w, self.hdd.w, decimal=10)
        
        mirrored = self.hdd.apply_function(lambda x: -x)
        np.testing.assert_array_almost_equal(mirrored.d, -self.hdd.d, decimal=10)
        np.testing.assert_array_almost_equal(mirrored.w, self.hdd.w, decimal=10)
    
    def test_reweigh_identity(self):
        f = lambda x: 2*np.ones(x.shape[0]) 
        twdNew = self.hdd.reweigh(f)
        self.assertIsInstance(twdNew, HypersphericalDiracDistribution)
        np.testing.assert_array_almost_equal(twdNew.d, self.hdd.d)
        np.testing.assert_array_almost_equal(twdNew.w, self.hdd.w)
        
    def test_reweigh_coord_based(self):
        f = lambda x: x[:, 1]
        twdNew = self.hdd.reweigh(f)
        self.assertIsInstance(twdNew, HypersphericalDiracDistribution)
        np.testing.assert_array_almost_equal(twdNew.d, self.hdd.d)
        self.assertAlmostEqual(np.sum(twdNew.w), 1, places=10)
        wNew = self.hdd.d[:, 1]*self.hdd.w
        np.testing.assert_array_almost_equal(twdNew.w, wNew/np.sum(wNew))
    
    def test_from_distribution(self):
        np.random.seed(0)
        vmf = VonMisesFisherDistribution(np.array([1, 1, 1])/np.sqrt(3), 1)
        dirac_dist = HypersphericalDiracDistribution.from_distribution(vmf, 100000)
        np.testing.assert_almost_equal(dirac_dist.mean_direction(), vmf.mean_direction(), decimal=2)

if __name__ == '__main__':
    unittest.main()
