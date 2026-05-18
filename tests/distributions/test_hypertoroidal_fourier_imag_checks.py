import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import abs as backend_abs
from pyrecest.backend import array, pi, real
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import (
    HypertoroidalFourierDistribution,
)


class HypertoroidalFourierImaginaryCheckTest(unittest.TestCase):
    def test_identity_pdf_discards_negligible_imaginary_roundoff(self):
        coeffs = array([1e-12j, 1.0 / (2.0 * pi), 0.0])
        dist = HypertoroidalFourierDistribution(coeffs, transformation="identity")

        npt.assert_allclose(dist.pdf(array([0.0])), array([1.0 / (2.0 * pi)]))

    def test_identity_pdf_rejects_large_imaginary_part(self):
        coeffs = array([-0.2j, 1.0 / (2.0 * pi), 0.0])
        dist = HypertoroidalFourierDistribution(coeffs, transformation="identity")

        with self.assertRaises(ValueError):
            dist.pdf(array([0.0]))

    def test_sqrt_pdf_uses_complex_modulus_squared(self):
        coeffs = array([0.2j, 1.0 / (2.0 * pi), 0.0])
        dist = HypertoroidalFourierDistribution(coeffs, transformation="sqrt")
        val = dist.value(array([0.0]))

        npt.assert_allclose(dist.pdf(array([0.0])), backend_abs(val) ** 2)
        self.assertGreater(float(backend_abs(val) ** 2), float(real(val) ** 2))


if __name__ == "__main__":
    unittest.main()