import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)


class HypertoroidalFourierImaginaryCheckTest(unittest.TestCase):
    def test_value_warns_for_large_negative_imaginary_part(self):
        coeffs = array([-0.2j, 1.0 / (2.0 * pi), 0.0])
        dist = HypertoroidalFourierDistribution(coeffs, transformation="identity")

        with self.assertWarns(RuntimeWarning):
            dist.value(array([0.0]))


if __name__ == "__main__":
    unittest.main()
