import unittest

import numpy as np
from pyrecest.distributions import HypertoroidalWNDistribution


class TestHypertoroidalWNDistribution(unittest.TestCase):
    def test_pdf(self):
        mu = np.array([[1], [2]])
        C = np.array([[0.5, 0.1], [0.1, 0.3]])

        hwn = HypertoroidalWNDistribution(mu, C)

        xa = np.array([[0, 1, 2], [1, 2, 3]]).T
        pdf_values = hwn.pdf(xa)

        expected_values = np.array(
            [0.0499028191873498, 0.425359477472412, 0.0499028191873498]
        )
        np.testing.assert_allclose(pdf_values, expected_values, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
