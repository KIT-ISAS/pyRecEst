"""Regression tests for camera projection geometry validation."""

import unittest

from pyrecest.backend import array, eye
from pyrecest.models import camera_projection_measurement


class TestCameraProjectionValidation(unittest.TestCase):
    def test_camera_projection_rejects_malformed_rotation_shape(self):
        state = array([1.0, 2.0, 4.0])

        invalid_rotations = (
            array([1.0, 0.0, 0.0]),
            array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        for rotation in invalid_rotations:
            with self.subTest(rotation_shape=tuple(rotation.shape)):
                with self.assertRaisesRegex(ValueError, "rotation"):
                    camera_projection_measurement(state, rotation=rotation)

    def test_camera_projection_rejects_malformed_camera_matrix_shape(self):
        state = array([1.0, 2.0, 4.0])

        invalid_camera_matrices = (
            array([1.0, 0.0, 0.0]),
            array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        for camera_matrix in invalid_camera_matrices:
            with self.subTest(camera_matrix_shape=tuple(camera_matrix.shape)):
                with self.assertRaisesRegex(ValueError, "camera_matrix"):
                    camera_projection_measurement(state, camera_matrix=camera_matrix)

    def test_camera_projection_accepts_well_formed_rotation_and_camera_matrix(self):
        state = array([2.0, 4.0, 2.0])
        camera_matrix = array([[2.0, 0.0, 10.0], [0.0, 3.0, 20.0], [0.0, 0.0, 1.0]])

        projected = camera_projection_measurement(
            state,
            rotation=eye(3),
            camera_matrix=camera_matrix,
        )

        self.assertEqual(tuple(projected.shape), (2,))


if __name__ == "__main__":
    unittest.main()
