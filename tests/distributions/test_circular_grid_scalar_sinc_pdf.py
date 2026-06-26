import unittest

from pyrecest.backend import array, pi
from pyrecest.distributions.circle.circular_grid_distribution import CircularGridDistribution


class CircularGridScalarSincPdfTest(unittest.TestCase):
    def test_pdf_via_sinc_accepts_scalar_angle(self):
        dist = CircularGridDistribution.from_function(
            lambda xs: xs * 0.0 + 1.0 / (2.0 * pi),
            9,
        )

        scalar_value = dist.pdf(0.0, use_sinc=True, sinc_repetitions=3)
        vector_value = dist.pdf(array([0.0]), use_sinc=True, sinc_repetitions=3)[0]

        self.assertAlmostEqual(float(scalar_value), float(vector_value))


if __name__ == "__main__":
    unittest.main()
