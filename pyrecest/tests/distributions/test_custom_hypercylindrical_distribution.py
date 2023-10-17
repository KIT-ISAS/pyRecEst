from math import pi
from pyrecest.backend import random
from pyrecest.backend import linspace
from pyrecest.backend import eye
from pyrecest.backend import array
import unittest

import numpy as np
from pyrecest.distributions import (
    GaussianDistribution,
    PartiallyWrappedNormalDistribution,
    VonMisesDistribution,
)
from pyrecest.distributions.cart_prod.custom_hypercylindrical_distribution import (
    CustomHypercylindricalDistribution,
)


class CustomHypercylindricalDistributionTest(unittest.TestCase):
    def setUp(self) -> None:
        mat = random.rand(6, 6)
        mat = mat @ mat.T
        self.pwn = PartiallyWrappedNormalDistribution(
            array([2, 3, 4, 5, 6, 7]), mat, 3
        )
        grid = np.mgrid[-3:4, -3:4, -2:3, -2:3, -2:3, -2:3]
        self.grid_flat = grid.reshape(6, -1).T

        self.vm = VonMisesDistribution(0, 1)
        self.gauss = GaussianDistribution(array([1, 2]), eye(2))

        def fun(x):
            return self.vm.pdf(x[:, 0]) * self.gauss.pdf(x[:, 1:])

        self.chcd_vm_gauss_stacked = CustomHypercylindricalDistribution(fun, 1, 2)

    def test_constructor(self):
        chd = CustomHypercylindricalDistribution(self.pwn.pdf, 3, 3)
        np.testing.assert_allclose(
            self.pwn.pdf(self.grid_flat), chd.pdf(self.grid_flat)
        )

    def test_from_distribution(self):
        chd = CustomHypercylindricalDistribution.from_distribution(self.pwn)
        np.testing.assert_allclose(
            self.pwn.pdf(self.grid_flat), chd.pdf(self.grid_flat)
        )

    def test_condition_on_linear(self):
        dist = self.chcd_vm_gauss_stacked.condition_on_linear([2, 1])

        x = linspace(0, 2 * pi, 100)
        np.testing.assert_allclose(dist.pdf(x), self.vm.pdf(x))

    def test_condition_on_periodic(self):
        dist = self.chcd_vm_gauss_stacked.condition_on_periodic(1)

        grid = np.mgrid[-3:4, -3:4].reshape(2, -1).T
        np.testing.assert_allclose(dist.pdf(grid), self.gauss.pdf(grid))


if __name__ == "__main__":
    unittest.main()