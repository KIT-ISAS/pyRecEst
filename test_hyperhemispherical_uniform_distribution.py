""" Test for uniform distribution for hyperhemispheres """
import unittest
import numpy as np
from hyperhemispherical_uniform_distribution import HyperhemisphericalUniformDistribution

class TestHyperhemisphericalUniformDistribution(unittest.TestCase):
    """ Test for uniform distribution for hyperhemispheres """
    def test_pdf_2d(self):
        hhud = HyperhemisphericalUniformDistribution(2)

        np.random.seed(10)
        points = np.random.randn(3, 100)
        points = points[:, points[2, :] >= 0]
        points /= np.linalg.norm(points, axis=0)

        self.assertTrue(np.allclose(hhud.pdf(points), np.ones(points.shape[1]) / (2 * np.pi), atol=1e-6))
    def test_pdf_3d(self):
        hhud = HyperhemisphericalUniformDistribution(3)

        np.random.seed(10)
        points = np.random.randn(4, 100)
        points = points[:, points[3, :] >= 0]
        points /= np.linalg.norm(points, axis=0)

        self.assertTrue(np.allclose(hhud.pdf(points), np.ones(points.shape[1]) / (np.pi**2), atol=1e-6))
    
    def test_integral_S2(self):
        hhud = HyperhemisphericalUniformDistribution(2)
        self.assertAlmostEqual(hhud.integral(), 1, delta=1e-6)

    def test_integral_S3(self):
        hhud = HyperhemisphericalUniformDistribution(3)
        self.assertAlmostEqual(hhud.integral(), 1, delta=1e-6)
    
if __name__ == '__main__':
    unittest.main()
