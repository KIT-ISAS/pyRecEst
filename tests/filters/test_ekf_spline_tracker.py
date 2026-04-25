import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, diag, eye, linalg
from pyrecest.filters import EKFSplineTracker, EkfSplineTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Finite-difference spline tracker tests are numpy-backend checks.",
)
class TestEKFSplineTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = EKFSplineTracker(
            kinematic_state=array([0.0, 0.0, 0.0, 0.0, 0.0]),
            scale_state=array([1.0, 1.0]),
            covariance=diag(array([0.1, 0.1, 0.01, 0.01, 0.01, 0.2, 0.2])),
            measurement_noise=0.01 * eye(2),
        )

    def test_initialization_and_alias(self):
        self.assertIs(EkfSplineTracker, EKFSplineTracker)
        self.assertEqual(self.tracker.state.shape, (7,))
        self.assertEqual(self.tracker.get_scaled_control_points().shape, (8, 2))
        self.assertEqual(
            self.tracker.get_point_estimate_extent(flatten_matrix=True).shape,
            (16,),
        )

    def test_contour_and_bounding_box_shapes(self):
        contour = self.tracker.get_contour_points(16)
        bounding_box = self.tracker.get_bounding_box(32)

        self.assertEqual(contour.shape, (16, 2))
        self.assertEqual(bounding_box["center_xy"].shape, (2,))
        self.assertEqual(bounding_box["dimension"].shape, (2,))
        self.assertGreater(float(bounding_box["dimension"][0]), 0.0)
        self.assertGreater(float(bounding_box["dimension"][1]), 0.0)

    def test_predict_uses_constant_turn_kinematics(self):
        tracker = EKFSplineTracker(
            kinematic_state=array([0.0, 0.0, 0.0, 2.0, 0.2]),
            covariance=diag(array([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])),
            measurement_noise=0.01 * eye(2),
        )

        tracker.predict(dt=0.5)

        npt.assert_allclose(
            tracker.kinematic_state[:3],
            array([1.0, 0.0, 0.1]),
            atol=1e-4,
        )
        self.assertTrue(np.all(linalg.eigvalsh(tracker.covariance) > -1e-10))

    def test_update_moves_position_and_scale_for_outer_measurement(self):
        tracker = EKFSplineTracker(
            covariance=diag(array([0.05, 0.05, 0.01, 0.01, 0.01, 0.2, 0.01])),
            measurement_noise=0.01 * eye(2),
        )

        tracker.update(array([3.0, 0.0]))

        self.assertGreater(float(tracker.kinematic_state[0]), 0.0)
        self.assertGreater(float(tracker.scale_state[0]), 1.0)
        self.assertIsNotNone(tracker.last_quadratic_form)
        self.assertTrue(np.all(linalg.eigvalsh(tracker.covariance) > -1e-10))

    def test_scale_correction_can_be_disabled(self):
        tracker = EKFSplineTracker(
            covariance=diag(array([0.05, 0.05, 0.01, 0.01, 0.01, 0.2, 0.2])),
            measurement_noise=0.01 * eye(2),
            scale_correction=False,
        )

        tracker.update(array([3.0, 0.0]))

        npt.assert_allclose(tracker.scale_state, array([1.0, 1.0]), atol=1e-10)

    def test_update_accepts_multiple_measurement_layouts(self):
        measurements = array([[3.0, 0.2], [0.0, 1.5], [-2.5, -0.1]])
        tracker = EKFSplineTracker(measurement_noise=0.01 * eye(2))

        tracker.update(measurements)
        tracker.update(measurements.T)

        self.assertIsNotNone(tracker.last_quadratic_form)


if __name__ == "__main__":
    unittest.main()
