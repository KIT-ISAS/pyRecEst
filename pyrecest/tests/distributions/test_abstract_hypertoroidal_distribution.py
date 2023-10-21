from math import pi
import unittest
from pyrecest.backend import array

from pyrecest.distributions import AbstractHypertoroidalDistribution
import numpy.testing as npt

class TestAbstractHypertoroidalDistribution(unittest.TestCase):
    def test_angular_error(self):
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(pi), array(0.0)), pi
        )
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(0), array(2 * pi)), 0
        )
        npt.assert_allclose(
            AbstractHypertoroidalDistribution.angular_error(array(pi / 4), array(7 * pi / 4)),
            pi / 2,
        )