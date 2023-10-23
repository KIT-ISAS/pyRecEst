from math import pi

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, concatenate, ones, sqrt, zeros

""" Test cases for DiskUniformDistribution"""
import unittest

import numpy.testing as npt
from pyrecest.distributions import DiskUniformDistribution


class TestDiskUniformDistribution(unittest.TestCase):
    """Test cases for DiskUniformDistribution"""

    def test_pdf(self):
        dist = DiskUniformDistribution()

        xs = array(
            [
                [0.5, 0, 1, 1 / sqrt(2), 0, 3, 1.5],
                [0.5, 1, 0, 1 / sqrt(2), 3, 0, 1.5],
            ]
        ).T
        pdf_values = dist.pdf(xs)

        npt.assert_allclose(
            pdf_values,
            1
            / pi
            * concatenate(
                (
                    ones(4),
                    zeros(3),
                )
            ),
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
