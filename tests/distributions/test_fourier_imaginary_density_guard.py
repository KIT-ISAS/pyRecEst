import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)


class FourierImaginaryDensityGuardTest(unittest.TestCase):
    def test_identity_pdf_rejects_negative_imaginary_density_value(self):
        coeffs = array([-0.2j, 1.0 / (2.0 * pi), 0.0], dtype=complex)

        dist = HypertoroidalFourierDistribution(coeffs, "identity")

        with self.assertRaisesRegex(ValueError, "non-negligible imaginary part"):
            dist.pdf(array([0.0]))


if __name__ == "__main__":
    unittest.main()
