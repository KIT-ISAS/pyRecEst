import unittest

import numpy as np
from pyrecest.distributions import BinghamDistribution

from .test_vmf_distribution import vectors_to_test_2d


class TestBinghamDistribution(unittest.TestCase):
    def test_pdf(self):
        M = np.array(
            [[1 / 3, 2 / 3, -2 / 3], [-2 / 3, 2 / 3, 1 / 3], [2 / 3, 1 / 3, 2 / 3]]
        )
        Z = np.array([-5, -3, 0])
        bd = BinghamDistribution(Z, M)

        np.testing.assert_array_almost_equal(
            bd.pdf(vectors_to_test_2d),
            np.array(
                [
                    0.0767812166360095,
                    0.0145020985787277,
                    0.0394207910410773,
                    0.0267197897401937,
                    0.0298598745474396,
                    0.0298598745474396,
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
