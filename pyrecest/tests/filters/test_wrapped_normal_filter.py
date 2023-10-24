import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.filters.wrapped_normal_filter import WrappedNormalFilter


class WrappedNormalFilterTest(unittest.TestCase):
    def setUp(self):
        self.wn_filter = WrappedNormalFilter()
        self.wn = WrappedNormalDistribution(array(1.3), array(0.8))
        self.meas_noise = WrappedNormalDistribution(array(0.0), array(0.9))
        self.wn_filter.filter_state = self.wn

    def test_initialization(self):
        wn1 = self.wn_filter.filter_state
        self.assertIsInstance(wn1, WrappedNormalDistribution)
        self.assertEqual(self.wn.mu, wn1.mu)
        self.assertEqual(self.wn.sigma, wn1.sigma)

    def test_predict_identity(self):
        self.wn_filter.predict_identity(
            WrappedNormalDistribution(array(0.0), self.wn.sigma)
        )
        wn_identity = self.wn_filter.filter_state
        self.assertIsInstance(wn_identity, WrappedNormalDistribution)
        self.assertEqual(self.wn.mu, wn_identity.mu)
        self.assertLess(self.wn.sigma, wn_identity.sigma)

    def test_update(self):
        # update identity
        self.wn_filter.update_identity(self.meas_noise, self.wn.mu)
        wn_identity = self.wn_filter.filter_state
        self.assertIsInstance(wn_identity, WrappedNormalDistribution)
        npt.assert_almost_equal(self.wn.mu, wn_identity.mu)
        self.assertGreater(self.wn.sigma, wn_identity.sigma)

        # reset filter state for the next test within this function
        self.wn_filter.filter_state = self.wn

        # update identity with different measurement
        self.wn_filter.update_identity(self.meas_noise, self.wn.mu + 0.1)
        wn_identity2 = self.wn_filter.filter_state
        self.assertIsInstance(wn_identity2, WrappedNormalDistribution)
        self.assertLess(self.wn.mu, wn_identity2.mu)
        self.assertGreater(self.wn.sigma, wn_identity2.sigma)


if __name__ == "__main__":
    unittest.main()
