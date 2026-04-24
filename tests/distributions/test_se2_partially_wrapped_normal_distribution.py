import unittest

import numpy.testing as npt

from pyrecest.backend import array
from pyrecest.distributions.cart_prod.se2_pwn_distribution import SE2PWNDistribution
from pyrecest.distributions.se2_partially_wrapped_normal_distribution import (
    SE2PartiallyWrappedNormalDistribution,
)


class TestSE2PartiallyWrappedNormalDistributionAlias(unittest.TestCase):
    def setUp(self):
        self.mu = array([1.0, 2.0, 3.0])
        self.C = array([[0.9, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])

    def test_alias_identity(self):
        self.assertIs(SE2PartiallyWrappedNormalDistribution, SE2PWNDistribution)

    def test_alias_exposes_pep8_helpers(self):
        dist = SE2PartiallyWrappedNormalDistribution(self.mu, self.C)
        npt.assert_allclose(dist.mean_4d(), dist.mean4D())
        npt.assert_allclose(dist.covariance_4d(), dist.covariance4D())

    def test_from_samples_returns_authoritative_class(self):
        dist = SE2PartiallyWrappedNormalDistribution(self.mu, self.C)
        fitted = SE2PartiallyWrappedNormalDistribution.from_samples(dist.sample(20))
        self.assertIsInstance(fitted, SE2PWNDistribution)


if __name__ == "__main__":
    unittest.main()
