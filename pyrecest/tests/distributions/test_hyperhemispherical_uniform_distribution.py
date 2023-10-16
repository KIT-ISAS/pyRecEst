from pyrecest.backend import reshape
from pyrecest.backend import ones
from pyrecest.backend import allclose
from pyrecest.backend import all
""" Test for uniform distribution for hyperhemispheres """
import unittest

import numpy as np
from pyrecest.distributions import HyperhemisphericalUniformDistribution


def get_random_points(n, d):
    np.random.seed(10)
    points = np.random.randn(n, d + 1)
    points = points[points[:, -1] >= 0, :]
    points /= reshape(np.linalg.norm(points, axis=1), (-1, 1))
    return points


class TestHyperhemisphericalUniformDistribution(unittest.TestCase):
    """Test for uniform distribution for hyperhemispheres"""

    def test_pdf_2d(self):
        hhud = HyperhemisphericalUniformDistribution(2)

        points = get_random_points(100, 2)

        self.assertTrue(
            allclose(
                hhud.pdf(points), ones(points.shape[0]) / (2 * np.pi), atol=1e-6
            )
        )

    def test_pdf_3d(self):
        hhud = HyperhemisphericalUniformDistribution(3)

        points = get_random_points(100, 3)
        # jscpd:ignore-start
        self.assertTrue(
            allclose(
                hhud.pdf(points), ones(points.shape[0]) / (np.pi**2), atol=1e-6
            )
        )
        # jscpd:ignore-end

    def test_integrate_S2(self):
        hhud = HyperhemisphericalUniformDistribution(2)
        self.assertAlmostEqual(hhud.integrate(), 1, delta=1e-6)

    def test_integrate_S3(self):
        hhud = HyperhemisphericalUniformDistribution(3)
        self.assertAlmostEqual(hhud.integrate(), 1, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
