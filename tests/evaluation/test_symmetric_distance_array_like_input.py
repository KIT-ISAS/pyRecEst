import unittest
from math import pi

from pyrecest.evaluation.get_distance_function import get_distance_function


class SymmetricDistanceArrayLikeInputTest(unittest.TestCase):
    def test_symmetric_circle_accepts_python_lists(self):
        distance = get_distance_function("circle", nSymm=2)
        self.assertAlmostEqual(float(distance([0.0], [pi])), 0.0)


if __name__ == "__main__":
    unittest.main()
