import unittest
from math import pi

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, mod
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)
from pyrecest.filters.toroidal_wrapped_normal_filter import ToroidalWrappedNormalFilter


class ToroidalWrappedNormalFilterTest(unittest.TestCase):
    def setUp(self):
        """Initial setup for each test."""
        self.mu = array([5.0, 2.5])
        self.C = array([[1.3, 1.4], [1.4, 2.0]])
        self.twn = ToroidalWrappedNormalDistribution(self.mu, self.C)

    def test_sanity_check(self):
        """Test setting and getting estimate of the filter."""
        curr_filter = ToroidalWrappedNormalFilter()
        curr_filter.filter_state = self.twn
        twn1 = curr_filter.filter_state
        self.assertIsInstance(twn1, ToroidalWrappedNormalDistribution)
        npt.assert_array_almost_equal(twn1.mu, self.twn.mu)
        npt.assert_array_almost_equal(twn1.C, self.twn.C)

    def test_predict_identity(self):
        """Test identity prediction of the filter."""
        curr_filter = ToroidalWrappedNormalFilter()
        curr_filter.filter_state = self.twn
        curr_filter.predict_identity(self.twn)
        dist_result = curr_filter.filter_state
        self.assertIsInstance(dist_result, ToroidalWrappedNormalDistribution)
        npt.assert_array_almost_equal(
            dist_result.mu, mod(self.twn.mu + self.twn.mu, 2 * pi)
        )
        npt.assert_array_almost_equal(dist_result.C, self.twn.C + self.twn.C)
