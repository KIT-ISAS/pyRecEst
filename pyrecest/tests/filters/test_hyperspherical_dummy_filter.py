import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.filters.hyperspherical_dummy_filter import HypersphericalDummyFilter


class HypersphericalDummyFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter_s2 = HypersphericalDummyFilter(2)
        self.filter_s3 = HypersphericalDummyFilter(3)

    def test_dim_s2(self):
        self.assertEqual(self.filter_s2.dim, 2)

    def test_dim_s3(self):
        self.assertEqual(self.filter_s3.dim, 3)

    def test_assert_dim_too_small(self):
        with self.assertRaises(AssertionError):
            HypersphericalDummyFilter(1)

    def test_filter_state_is_uniform(self):
        self.assertIsInstance(
            self.filter_s2.filter_state, HypersphericalUniformDistribution
        )

    def test_get_point_estimate_unit_norm_s2(self):
        est = self.filter_s2.get_point_estimate()
        self.assertEqual(est.shape, (3,))
        npt.assert_allclose(linalg.norm(est), 1.0, atol=1e-10)

    def test_get_point_estimate_unit_norm_s3(self):
        est = self.filter_s3.get_point_estimate()
        self.assertEqual(est.shape, (4,))
        npt.assert_allclose(linalg.norm(est), 1.0, atol=1e-10)

    def test_predict_identity_is_noop(self):
        noise = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 1.0)
        state_before = self.filter_s2.filter_state
        self.filter_s2.predict_identity(noise)
        self.assertIs(self.filter_s2.filter_state, state_before)

    def test_predict_nonlinear_is_noop(self):
        state_before = self.filter_s2.filter_state
        self.filter_s2.predict_nonlinear(lambda x: x)
        self.assertIs(self.filter_s2.filter_state, state_before)

    def test_update_identity_is_noop(self):
        noise = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 1.0)
        measurement = array([0.0, 0.0, 1.0])
        state_before = self.filter_s2.filter_state
        self.filter_s2.update_identity(noise, measurement)
        self.assertIs(self.filter_s2.filter_state, state_before)

    def test_update_nonlinear_is_noop(self):
        state_before = self.filter_s2.filter_state
        self.filter_s2.update_nonlinear(lambda z, x: x)
        self.assertIs(self.filter_s2.filter_state, state_before)

    def test_filter_state_setter_is_noop(self):
        original_state = self.filter_s2.filter_state
        new_dist = HypersphericalUniformDistribution(2)
        self.filter_s2.filter_state = new_dist
        self.assertIs(self.filter_s2.filter_state, original_state)

    def test_set_state_is_noop(self):
        original_state = self.filter_s2.filter_state
        new_dist = HypersphericalUniformDistribution(2)
        self.filter_s2.set_state(new_dist)
        self.assertIs(self.filter_s2.filter_state, original_state)

    def test_get_estimate_returns_distribution(self):
        est = self.filter_s2.get_estimate()
        self.assertIsInstance(est, HypersphericalUniformDistribution)


if __name__ == "__main__":
    unittest.main()
