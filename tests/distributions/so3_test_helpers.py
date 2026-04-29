"""Shared test helpers for SO(3) distributions."""

import math

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, eye, linalg, pi, sin, sqrt, to_numpy

ATOL = 1e-6


def scalar(value):
    """Return a backend scalar as a Python float."""
    return float(to_numpy(value).reshape(-1)[0])


def z_quaternion(angle):
    """Return a scalar-last quaternion for a rotation around the z axis."""
    return array([0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)])


def x_quaternion(angle):
    """Return a scalar-last quaternion for a rotation around the x axis."""
    return array([sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)])


def z_rotation(angle):
    """Return a rotation matrix for a rotation around the z axis."""
    return array(
        [
            [cos(angle), -sin(angle), 0.0],
            [sin(angle), cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def assert_matches_z_rotation(test_case, rotation_matrix, angle):
    """Assert that a matrix is the expected z rotation and is orthogonal."""
    npt.assert_allclose(rotation_matrix, z_rotation(angle), atol=ATOL)
    npt.assert_allclose(rotation_matrix.T @ rotation_matrix, eye(3), atol=ATOL)
    test_case.assertEqual(rotation_matrix.shape, (3, 3))


def assert_pdf_peak_matches_log_pdf(test_case, dist, covariance, tangent_dim, offset):
    """Assert that a tangent Gaussian peaks at its mode with matching log density."""
    mode_pdf = scalar(dist.pdf(dist.mode()))
    offset_pdf = scalar(dist.pdf(offset))
    expected_mode_pdf = 1.0 / scalar(
        sqrt((2.0 * pi) ** tangent_dim * linalg.det(covariance))
    )

    test_case.assertGreater(mode_pdf, offset_pdf)
    npt.assert_allclose(mode_pdf, expected_mode_pdf, atol=ATOL)
    npt.assert_allclose(
        scalar(dist.ln_pdf(dist.mode())), math.log(mode_pdf), atol=ATOL
    )
