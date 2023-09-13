
import unittest

import numpy as np
from pyrecest.distributions.hypersphere_subset.hemispherical_uniform_distribution import HemisphericalUniformDistribution
from pyrecest.tests.distributions.test_hyperhemispherical_uniform_distribution import get_random_points

class TestHyperhemisphericalUniformDistribution(unittest.TestCase):
    """Test for uniform distribution for hyperhemispheres"""

    def test_pdf_2d(self):
        hhud = HemisphericalUniformDistribution()

        points = get_random_points(100, 2)

        # jscpd:ignore-start
        self.assertTrue(
            np.allclose(
                hhud.pdf(points), np.ones(points.shape[0]) / (2 * np.pi), atol=1e-6
            )
        )
        # jscpd:ignore-end