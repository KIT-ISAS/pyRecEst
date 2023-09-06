import unittest

import numpy as np

from pyrecest.distributions import AbstractHypertoroidalDistribution


class TestAbstractHypertoroidalDistribution(unittest.TestCase):
    def test_angular_error(self):
        np.testing.assert_allclose(AbstractHypertoroidalDistribution.angular_error(np.pi, 0), np.pi)
        np.testing.assert_allclose(AbstractHypertoroidalDistribution.angular_error(0, 2 * np.pi), 0)
        np.testing.assert_allclose(AbstractHypertoroidalDistribution.angular_error(np.pi / 4, 7 * np.pi / 4), np.pi/2)
