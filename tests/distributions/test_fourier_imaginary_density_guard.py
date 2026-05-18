import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi, zeros
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)


class FourierImaginaryDensityGuardTest(unittest.TestCase):
    def test_identity_pdf_rejects_negative_imaginary_density_value(self):
        coeffs = zeros((3,), dtype=complex)
        coeffs[0] = -0.2j
        coeffs[1] = 1.0 / (2.0 * pi)

        dist = HypertoroidalFourierDistribution(coeffs, "identity")

        with self.assertRaisesRegex(ValueError, "non-negligible imaginary part"):
            dist.pdf(array([0.0]))


if __name__ == "__main__":
    unittest.main()
