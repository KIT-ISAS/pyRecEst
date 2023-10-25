import unittest
from math import pi

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, arange, array, column_stack, diff, ones, zeros
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)


class AbstractHypercylindricalDistributionTest(unittest.TestCase):
    def test_mode_numerical_gaussian_2D(self):
        mu = array([5.0, 1.0])
        C = array([[2.0, 1.0], [1.0, 1.0]])
        g = PartiallyWrappedNormalDistribution(mu, C, 1)
        self.assertTrue(allclose(g.mode_numerical(), mu, atol=1e-5))

    def test_linear_mean_numerical(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        npt.assert_allclose(hwn.linear_mean_numerical(), hwn.mu[-1])

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.pytorch",
        reason="Not supported on PyTorch backend",
    )
    def test_condition_on_periodic(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        dist_cond1 = hwn.condition_on_periodic(array(1.5))
        # There is some normalization constant involved, therefore, test if ratio stays the same
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([1.5 * ones(11), arange(-5, 6)]))
                / dist_cond1.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )
        dist_cond2 = hwn.condition_on_periodic(array(1.5) + 2.0 * pi)
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([1.5 * ones(11), arange(-5, 6)]))
                / dist_cond2.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )

    @unittest.skipIf(
        pyrecest.backend.__name__ == "pyrecest.pytorch",
        reason="Not supported on PyTorch backend",
    )
    def test_condition_on_linear(self):
        hwn = PartiallyWrappedNormalDistribution(
            array([1.0, 2.0]), array([[2.0, 0.3], [0.3, 1.0]]), 1
        )
        dist_cond1 = hwn.condition_on_linear(array(1.5))
        npt.assert_allclose(
            diff(
                hwn.pdf(column_stack([arange(-5, 6), 1.5 * ones(11)]))
                / dist_cond1.pdf(arange(-5, 6))
            ),
            zeros(10),
            atol=1e-10,
        )
        dist_cond2 = hwn.condition_on_linear(array(1.5 + 2.0 * pi))
        self.assertFalse(
            (
                allclose(
                    diff(
                        hwn.pdf(column_stack([arange(-5, 6), 1.5 * ones(11)]))
                        / dist_cond2.pdf(arange(-5, 6))
                    ),
                    zeros(10),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
