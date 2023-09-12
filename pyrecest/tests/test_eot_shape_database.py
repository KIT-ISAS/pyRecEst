import unittest

from pyrecest.evaluation.eot_shape_database import Star, StarShapedPolygon
from shapely.geometry import Polygon


class TestStarConvexPolygon(unittest.TestCase):
    def setUp(self):
        self.square_star_convex_poly = StarShapedPolygon(
            [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)]
        )

    def test_area(self):
        square_poly = Polygon([(-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)])
        self.assertEqual(self.square_star_convex_poly.area, square_poly.area)

    def test_is_convex(self):
        self.assertTrue(self.square_star_convex_poly.is_convex())

    def test_compute_kernel_convex(self):
        self.assertEqual(
            self.square_star_convex_poly.compute_kernel().area,
            self.square_star_convex_poly.area,
        )

    def test_compute_kernel_star_convex(self):
        star_full = Star(0.5)
        star_kernel = star_full.compute_kernel()
        # Kernel no more than 50% of the area of the original polygon
        self.assertLess(star_kernel.area, 0.5 * star_full.area)


if __name__ == "__main__":
    unittest.main()
