import math
import unittest

from pyrecest.evaluation.get_distance_function import get_distance_function


class SymmetricDistanceArrayLikeInputTest(unittest.TestCase):
    def test_symmetric_circle_accepts_array_like_truth_values(self):
        distance = get_distance_function("circle", nSymm=2)

        self.assertAlmostEqual(float(distance([0.0], [math.pi])), 0.0)


if __name__ == "__main__":
    unittest.main()
