""" Test cases for DiskUniformDistribution"""
import unittest
import numpy as np
from pyrecest.distributions import DiskUniformDistribution

class TestDiskUniformDistribution(unittest.TestCase):
    """ Test cases for DiskUniformDistribution"""

    def test_pdf(self):
        dist = DiskUniformDistribution()

        xs = np.array([[0.5, 0, 1, 1/np.sqrt(2), 0, 3, 1.5],
                      [0.5, 1, 0, 1/np.sqrt(2), 3, 0, 1.5]]).T
        pdf_values = dist.pdf(xs)

        np.testing.assert_allclose(
            pdf_values, 1/np.pi * np.concatenate((np.ones(4), np.zeros(3),)), rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
