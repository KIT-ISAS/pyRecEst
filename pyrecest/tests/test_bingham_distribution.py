import unittest
import numpy as np
from bingham_distribution import BinghamDistribution


class TestBinghamDistribution(unittest.TestCase):
    def test_pdf(self):
        M = np.array([[1/3, 2/3, -2/3], [-2/3, 2/3, 1/3], [2/3, 1/3, 2/3]])
        Z = np.array([-5, -3, 0])
        bd = BinghamDistribution(Z, M)

        xs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]/np.sqrt(2), [1, 1, 2] /
                    np.linalg.norm([1, 1, 2]), -np.array([1, 1, 2])/np.linalg.norm([1, 1, 2])])
        
        np.testing.assert_array_almost_equal(
            bd.pdf(xs),
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
