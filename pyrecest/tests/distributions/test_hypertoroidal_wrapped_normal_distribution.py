from pyrecest.backend import array
import unittest
import numpy.testing as npt

from pyrecest.distributions import HypertoroidalWNDistribution


class TestHypertoroidalWNDistribution(unittest.TestCase):
    def test_pdf(self):
        mu = array([[1], [2]])
        C = array([[0.5, 0.1], [0.1, 0.3]])

        hwn = HypertoroidalWNDistribution(mu, C)

        xa = array([[0, 1, 2], [1, 2, 3]]).T
        pdf_values = hwn.pdf(xa)

        expected_values = array(
            [0.0499028191873498, 0.425359477472412, 0.0499028191873498]
        )
        npt.assert_allclose(pdf_values, expected_values, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()