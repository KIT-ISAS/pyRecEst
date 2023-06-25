import unittest
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.filters.wrapped_normal_filter import WrappedNormalFilter
import numpy as np

class WrappedNormalFilterTest(unittest.TestCase):
    
    def test_initialization(self):
        wn_filter = WrappedNormalFilter()
        wn = WrappedNormalDistribution(1.3, 0.8)
        
        # Sanity check
        wn_filter.filter_state = wn
        wn1 = wn_filter.filter_state
        self.assertIsInstance(wn1, WrappedNormalDistribution)
        self.assertEqual(wn.mu, wn1.mu)
        self.assertEqual(wn.sigma, wn1.sigma)

    def test_predict_identity(self):
        wn_filter = WrappedNormalFilter()
        wn = WrappedNormalDistribution(1.3, 0.8)

        wn_filter.filter_state = wn
        wn_filter.predict_identity(WrappedNormalDistribution(0, wn.sigma))
        wn_identity = wn_filter.filter_state
        self.assertIsInstance(wn_identity, WrappedNormalDistribution)
        self.assertEqual(wn.mu, wn_identity.mu)
        self.assertLess(wn.sigma, wn_identity.sigma)

    def test_update(self):
        wn_filter = WrappedNormalFilter()
        wn = WrappedNormalDistribution(1.3, 0.8)
        meas_noise = WrappedNormalDistribution(0, 0.9)

        # update identity
        wn_filter.filter_state = wn
        wn_filter.update_identity(meas_noise, wn.mu)
        wn_identity = wn_filter.filter_state
        self.assertIsInstance(wn_identity, WrappedNormalDistribution)
        np.testing.assert_almost_equal(wn.mu, wn_identity.mu)
        self.assertGreater(wn.sigma, wn_identity.sigma)
        
        # update identity with different measurement
        wn_filter.filter_state = wn
        wn_filter.update_identity(meas_noise, wn.mu + 0.1)
        wn_identity2 = wn_filter.filter_state
        self.assertIsInstance(wn_identity2, WrappedNormalDistribution)
        self.assertLess(wn.mu, wn_identity2.mu)
        self.assertGreater(wn.sigma, wn_identity2.sigma)

if __name__ == '__main__':
    unittest.main()
