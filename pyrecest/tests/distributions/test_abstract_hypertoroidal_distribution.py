from math import pi
import unittest
from pyrecest.backend import array
import numpy as np
from pyrecest.distributions import AbstractHypertoroidalDistribution


class TestAbstractHypertoroidalDistribution(unittest.TestCase):
    def test_angular_error(self):
        np.testing.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(pi), array(0.0)), pi
        )
        np.testing.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(0), array(2 * pi)), 0
        )
        np.testing.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(pi / 4), array(7 * pi / 4)),
            pi / 2,
        )