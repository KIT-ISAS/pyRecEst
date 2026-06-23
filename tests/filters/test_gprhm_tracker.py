import unittest

import numpy as np
from pyrecest.backend import array, empty, eye, vstack
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.eot_shape_database import StarFish
from pyrecest.filters.gprhm_tracker import GPRHMTracker
from pyrecest.utils.metrics import iou_polygon
from shapely.geometry import Polygon


def random_points_on_contour(shape, n_points):
    points = np.empty((0,), dtype=object)
    for _ in range(n_points):
        # Get the boundary of the polygon
        boundary = shape.boundary
        # Get the total length of the boundary
        total_length = boundary.length
        # Generate random distances along the boundary
        sample_distances = np.random.uniform(
            low=0.0, high=total_length, size=(n_points,)
        )
        # Get the points on the boundary at these distances
        points = np.hstack(
            (
                points,
                np.array(
                    [boundary.interpolate(distance) for distance in sample_distances],
                    dtype=object,
                ),
            )
        )
    return points


class TestGPRHMTracker(unittest.TestCase):
    def setUp(self):
        # Assume initialization code for the tracker and other components is here
        self.tracker = GPRHMTracker(
            n_base_points=25,
            kernel_params=[2.0, np.pi / 4],
            log_prior_estimates=True,
            log_posterior_estimates=True,
            log_prior_extents=True,
            log_posterior_extents=True,
        )
        self.gt_shape = StarFish()
        self.n_steps = 5
        self.n_measurements = 10
        self.R = eye(2) * 0.1  # Measurement noise covariance

    def test_constructor_accepts_scalar_integer_base_point_count(self):
        tracker = GPRHMTracker(
            n_base_points=np.array(8),
            log_prior_estimates=False,
            log_posterior_estimates=False,
            log_prior_extents=False,
            log_posterior_extents=False,
        )

        self.assertEqual(tracker.phi_pts.shape, (8,))

    def test_constructor_rejects_invalid_base_point_counts(self):
        for invalid_count in (0, True, 8.5, "8", [8]):
            with self.subTest(n_base_points=invalid_count), self.assertRaises(
                ValueError
            ):
                GPRHMTracker(n_base_points=invalid_count)

    def test_get_contour_points_honors_requested_count(self):
        requested_points = 32

        predicted_points = self.tracker.get_contour_points(requested_points)

        self.assertEqual(predicted_points.shape, (requested_points, 2))

    def test_grid_and_contour_counts_must_be_positive_integers(self):
        for invalid_count in (0, True, 5.5, "5", [5]):
            with self.subTest(method="grid", n=invalid_count), self.assertRaises(
                ValueError
            ):
                self.tracker.get_extents_on_grid(invalid_count)
            with self.subTest(method="contour", n=invalid_count), self.assertRaises(
                ValueError
            ):
                self.tracker.get_contour_points(invalid_count)

    def test_iou_after_updates(self):
        for _ in range(self.n_steps):
            points_from_contour_no_noise = random_points_on_contour(
                shape=self.gt_shape, n_points=self.n_measurements
            )
            measurements = empty((0, 2))
            for curr_point in points_from_contour_no_noise:
                curr_meas = GaussianDistribution(
                    array(curr_point.coords).flatten(), self.R
                ).sample(1)
                measurements = vstack((measurements, curr_meas))

            for curr_meas in measurements:
                self.tracker.update(curr_meas, R=self.R)

        # Assuming get_contour_points method returns a list of (x, y) tuples
        predicted_points = self.tracker.get_contour_points(100)
        predicted_polygon = Polygon(predicted_points)

        gt_points = self.gt_shape.exterior.coords
        gt_polygon = Polygon(gt_points)

        iou = iou_polygon(predicted_polygon, gt_polygon)
        self.assertTrue(iou > 0.85, f"IOU was {iou}, but expected > 0.6")


# Running the test
if __name__ == "__main__":
    unittest.main()
