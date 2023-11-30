import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, cos, sin
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.filters.von_mises_fisher_filter import VonMisesFisherFilter


class VMFFilterTest(unittest.TestCase):
    def setUp(self):
        """Initial setup for each test."""
        self.filter = VonMisesFisherFilter()
        self.phi = 0.3
        self.mu = array([cos(self.phi), sin(self.phi)])
        self.kappa = 0.7
        self.vmf = VonMisesFisherDistribution(self.mu, self.kappa)

    def test_VMFFilter2d(self):
        """Test VonMisesFisherFilter in 2d."""
        self.filter.filter_state = self.vmf
        vmf_result = self.filter.filter_state
        self.assertEqual(type(vmf_result), VonMisesFisherDistribution)
        self.assertTrue(allclose(self.vmf.mu, vmf_result.mu))
        self.assertEqual(self.vmf.kappa, vmf_result.kappa)

    def test_prediction_identity(self):
        """Test prediction identity."""
        self.filter.state = self.vmf
        noise_distribution = VonMisesFisherDistribution(array([0.0, 1.0]), 0.9)
        self.filter.predict_identity(noise_distribution)
        # Add other assertions and logic here

    def test_update_identity(self):
        """Test update identity."""
        self.filter.filter_state = self.vmf
        noise_distribution = VonMisesFisherDistribution(array([0.0, 1.0]), 0.9)
        self.filter.update_identity(noise_distribution, self.vmf.mu)
        vmf_updated_identity = self.filter.filter_state
        self.assertEqual(type(vmf_updated_identity), VonMisesFisherDistribution)
        npt.assert_allclose(self.vmf.mu, vmf_updated_identity.mu, rtol=5e-7)
        self.assertGreaterEqual(vmf_updated_identity.kappa, self.vmf.kappa)


if __name__ == "__main__":
    unittest.main()
