import unittest

import numpy as np
from pyrecest.distributions.cart_prod.hypercylindrical_partially_wrapped_normal_distribution import (
    HypercylindricalPartiallyWrappedNormalDistribution,
)


class AbstractHypercylindricalDistributionTest(unittest.TestCase):
    def test_mode_numerical_gaussian_2D(self):
        mu = np.array([5, 1])
        C = np.array([[2, 1], [1, 1]])
        g = HypercylindricalPartiallyWrappedNormalDistribution(mu, C, 1)
        self.assertTrue(np.allclose(g.mode_numerical(), mu, atol=1e-5))

    def test_linear_mean_numerical(self):
        hwn = HypercylindricalPartiallyWrappedNormalDistribution(
            np.array([1, 2]), np.array([[2, 0.3], [0.3, 1]]), 1
        )
        np.testing.assert_allclose(hwn.linear_mean_numerical(), hwn.mu[-1])

    def test_condition_on_periodic(self):
        hwn = HypercylindricalPartiallyWrappedNormalDistribution(
            np.array([1, 2]), np.array([[2, 0.3], [0.3, 1]]), 1
        )
        dist_cond1 = hwn.condition_on_periodic(np.array(1.5))
        # There is some normalization constant involved, therefore, test if ratio stays the same
        np.testing.assert_allclose(
            np.diff(
                hwn.pdf(np.column_stack([1.5 * np.ones(11), np.arange(-5, 6)]))
                / dist_cond1.pdf(np.arange(-5, 6))
            ),
            np.zeros(10),
            atol=1e-10,
        )
        dist_cond2 = hwn.condition_on_periodic(np.array(1.5) + 2 * np.pi)
        np.testing.assert_allclose(
            np.diff(
                hwn.pdf(np.column_stack([1.5 * np.ones(11), np.arange(-5, 6)]))
                / dist_cond2.pdf(np.arange(-5, 6))
            ),
            np.zeros(10),
            atol=1e-10,
        )

    def test_condition_on_linear(self):
        hwn = HypercylindricalPartiallyWrappedNormalDistribution(
            np.array([1, 2]), np.array([[2, 0.3], [0.3, 1]]), 1
        )
        dist_cond1 = hwn.condition_on_linear(np.array(1.5))
        np.testing.assert_allclose(
            np.diff(
                hwn.pdf(np.column_stack([np.arange(-5, 6), 1.5 * np.ones(11)]))
                / dist_cond1.pdf(np.arange(-5, 6))
            ),
            np.zeros(10),
            atol=1e-10,
        )
        dist_cond2 = hwn.condition_on_linear(np.array(1.5 + 2 * np.pi))
        self.assertFalse(
            (
                np.allclose(
                    np.diff(
                        hwn.pdf(np.column_stack([np.arange(-5, 6), 1.5 * np.ones(11)]))
                        / dist_cond2.pdf(np.arange(-5, 6))
                    ),
                    np.zeros(10),
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
