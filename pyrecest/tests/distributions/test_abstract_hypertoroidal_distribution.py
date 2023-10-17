from math import pi
import unittest

import numpy as np
from pyrecest.distributions import AbstractHypertoroidalDistribution


class TestAbstractHypertoroidalDistribution(unittest.TestCase):
    def test_angular_error(self):
        np.testing.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(pi, 0), pi
        )
        np.testing.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(0, 2 * pi), 0
        )
        np.testing.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(pi / 4, 7 * pi / 4),
            pi / 2,
        )