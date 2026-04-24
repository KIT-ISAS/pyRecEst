import unittest

import numpy.testing as npt
from parameterized import parameterized

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, column_stack, linalg, pi
from pyrecest.distributions.hypersphere_subset.abstract_sphere_subset_distribution import (
    AbstractSphereSubsetDistribution,
)


class TestAbstractSphereSubsetDistribution(unittest.TestCase):
    @parameterized.expand(
        [
            ("colatitude",),
            ("elevation",),
        ]
    )
    def test_cart_to_sph_to_cart(self, mode):
        # jscpd:ignore-start
        # Create some Cartesian coordinates
        x = array([1.0, 0.0, 0.0])
        y = array([0.0, 1.0, 0.0])
        z = array([0.0, 0.0, 1.0])
        # jscpd:ignore-end

        # Convert to spherical coordinates and back
        azimuth, theta = AbstractSphereSubsetDistribution.cart_to_sph(
            x, y, z, mode=mode
        )
        x_new, y_new, z_new = AbstractSphereSubsetDistribution.sph_to_cart(
            azimuth, theta, mode=mode
        )

        # The new Cartesian coordinates should be close to the original ones
        npt.assert_allclose(x_new, x, atol=1e-7)
        npt.assert_allclose(y_new, y, atol=1e-7)
        npt.assert_allclose(z_new, z, atol=1e-7)

    @parameterized.expand(
        [
            ("colatitude",),
            ("elevation",),
        ]
    )
    def test_sph_to_cart_to_sph(self, mode):
        # Create some spherical coordinates. Warning! For angles of, 0,
        # the transformation from spherical to Cartesian coordinates may not be
        # uniquely invertible.
        azimuth = array([0.5, 0.1, pi / 4, pi / 2, 1])
        theta = array([0.2, pi / 2 - 0.1, pi / 4, 0.1, 1])
        assert not any(
            (azimuth == 0.0) | (theta == 0.0)
        ), "Do not include tests with one of the two angles 0 because the conversion may not be uniquely invertible"

        # Convert to Cartesian coordinates and back
        x, y, z = AbstractSphereSubsetDistribution.sph_to_cart(
            azimuth, theta, mode=mode
        )
        npt.assert_allclose(linalg.norm(column_stack((x, y, z)), axis=1), 1)
        azimuth_new, theta_new = AbstractSphereSubsetDistribution.cart_to_sph(
            x, y, z, mode=mode
        )

        # The new spherical coordinates should be close to the original ones
        npt.assert_allclose(azimuth_new, azimuth, rtol=5e-6)
        npt.assert_allclose(theta_new, theta, rtol=5e-6)

    def test_sph_to_cart_to_sph_ang_0_colatitude(self):
        # Create some spherical coordinates. Warning! For angles of, 0,
        # the transformation from spherical to Cartesian coordinates may not be
        # uniquely invertible.
        azimuth = array([0.0, 0.0, 0.0])
        theta = array([0.0, pi / 2 - 0.1, pi / 2])

        # Convert to Cartesian coordinates and back
        x, y, z = AbstractSphereSubsetDistribution.sph_to_cart(
            azimuth, theta, mode="colatitude"
        )
        npt.assert_allclose(linalg.norm(column_stack((x, y, z)), axis=1), 1)
        azimuth_new, theta_new = AbstractSphereSubsetDistribution.cart_to_sph(
            x, y, z, mode="colatitude"
        )

        # The new spherical coordinates should be close to the original ones
        npt.assert_allclose(azimuth_new, array([0, 0, 0]), atol=1e-15)
        npt.assert_allclose(theta_new, array([pi / 2, pi / 2, pi / 2]), atol=1e-15)
