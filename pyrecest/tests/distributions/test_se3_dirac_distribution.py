from pyrecest.backend import linalg
from pyrecest.backend import tile
from pyrecest.backend import sum
from pyrecest.backend import concatenate
from pyrecest.backend import array
import unittest

import numpy as np
from pyrecest.distributions import (
    GaussianDistribution,
    HyperhemisphericalUniformDistribution,
)
from pyrecest.distributions.se3_cart_prod_stacked_distribution import (
    SE3CartProdStackedDistribution,
)
from pyrecest.distributions.se3_dirac_distribution import SE3DiracDistribution


class SE3DiracDistributionTest(unittest.TestCase):
    def test_constructor(self):
        dSph = array(
            [
                [1, 2, 3, 4, 5, 6],
                [2, 4, 0, 0.5, 1, 1],
                [5, 10, 20, 30, 40, 50],
                [2, 31, 42, 3, 9.9, 5],
            ]
        ).T
        dSph = dSph / linalg.norm(dSph, axis=-1, keepdims=True)
        dLin = tile(array([-5, 0, 5, 10, 15, 20]), (3, 1)).T
        w = array([1, 2, 3, 1, 2, 3])
        w = w / sum(w)
        SE3DiracDistribution(concatenate((dSph, dLin), axis=-1), w)

    def test_from_distribution(self):
        cpsd = SE3CartProdStackedDistribution(
            [
                HyperhemisphericalUniformDistribution(3),
                GaussianDistribution(array([1, 2, 3]).T, np.diag([3, 2, 1])),
            ]
        )
        SE3DiracDistribution.from_distribution(cpsd, 100)


if __name__ == "__main__":
    unittest.main()