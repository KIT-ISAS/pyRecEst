import unittest
from math import pi

import numpy.testing as npt
from parameterized import parameterized
from pyrecest.backend import array
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
        # Create some Cartesian coordinates
        x = array([1.0, 0.0, 0.0])
        y = array([0.0, 1.0, 0.0])
        z = array([0.0, 0.0, 1.0])

        # Convert to spherical coordinates and back
        azimuth, theta = AbstractSphereSubsetDistribution.cart_to_sph(
            x, y, z, mode=mode
        )
        x_new, y_new, z_new = AbstractSphereSubsetDistribution.sph_to_cart(
            azimuth, theta, mode=mode
        )

        # The new Cartesian coordinates should be close to the original ones
        npt.assert_allclose(x_new, x, atol=1e-15)
        npt.assert_allclose(y_new, y, atol=1e-15)
        npt.assert_allclose(z_new, z, atol=1e-15)

    @parameterized.expand(
        [
            ("colatitude",),
            ("elevation",),
        ]
    )
    def test_sph_to_cart_to_sph(self, mode):
        # Create some spherical coordinates. Do *not* use 0 as theta because
        # the transformation from spherical to Cartesian coordinates is not
        # uniquely invertible in this case.
        azimuth = array([0.0, pi / 4, pi / 2])
        theta = array([pi / 2, pi / 4, 0.1])

        # Convert to Cartesian coordinates and back
        x, y, z = AbstractSphereSubsetDistribution.sph_to_cart(
            azimuth, theta, mode=mode
        )
        azimuth_new, theta_new = AbstractSphereSubsetDistribution.cart_to_sph(
            x, y, z, mode=mode
        )

        # The new spherical coordinates should be close to the original ones
        npt.assert_allclose(azimuth_new, azimuth, atol=1e-15)
        npt.assert_allclose(theta_new, theta, atol=1e-15)
