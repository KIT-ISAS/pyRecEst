from math import pi
from pyrecest.backend import where
from pyrecest.backend import sin
from pyrecest.backend import ndim
from pyrecest.backend import cos
from pyrecest.backend import arctan2
from pyrecest.backend import arccos

from beartype import beartype

from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractSphereSubsetDistribution(AbstractHypersphereSubsetDistribution):
    """
    This is an abstract class for a distribution over a sphere.
    """

    def __init__(self):
        """
        Initialize the AbstractSphereSubsetDistribution instance.
        """
        super().__init__(2)

    @staticmethod
    def sph_to_cart(phi, theta, mode="colatitude") -> tuple:
        """
        Convert spherical coordinates to Cartesian coordinates.

        Args:
            phi (): Azimuth angles.
            theta (): Colatitude angles or elevation angles based on the mode.
            mode (str): Either 'colatitude' or 'elevation'.

        Returns:
            tuple: Cartesian coordinates.
        """
        assert ndim(phi) == 1 and ndim(theta) == 1, "Inputs must be 1-dimensional"
        if mode == "colatitude":
            x, y, z = AbstractSphereSubsetDistribution._sph_to_cart_colatitude(
                phi, theta
            )
        elif mode == "elevation":
            x, y, z = AbstractSphereSubsetDistribution._sph_to_cart_elevation(
                phi, theta
            )
        else:
            raise ValueError("Mode must be either 'colatitude' or 'elevation'")

        return x, y, z

    @staticmethod
    def cart_to_sph(
        x, y, z, mode="colatitude"
    ) -> tuple:
        """
        Convert Cartesian coordinates to spherical coordinates.

        Args:
            x (): X coordinates.
            y (): Y coordinates.
            z (): Z coordinates.

        Returns:
            tuple: Spherical coordinates.
        """
        assert (
            ndim(x) == 1 and ndim(y) == 1 and ndim(z)
        ), "Inputs must be 1-dimensional"
        if mode == "colatitude":
            phi, theta = AbstractSphereSubsetDistribution._cart_to_sph_colatitude(
                x, y, z
            )
        elif mode == "elevation":
            phi, theta = AbstractSphereSubsetDistribution._cart_to_sph_elevation(
                x, y, z
            )
        else:
            raise ValueError("Mode must be either 'colatitude' or 'elevation'")

        return phi, theta

    @staticmethod
    def _sph_to_cart_colatitude(azimuth, colatitude) -> tuple:
        assert ndim(azimuth) == 1 and ndim(
            colatitude
        ), "Inputs must be 1-dimensional"
        x = sin(colatitude) * cos(azimuth)
        y = sin(colatitude) * sin(azimuth)
        z = cos(colatitude)
        return x, y, z

    @staticmethod
    def _sph_to_cart_elevation(azimuth, elevation) -> tuple:
        """
        Convert spherical coordinates (using elevation) to Cartesian coordinates.
        Assumes a radius of 1.

        Args:
            azimuth (): Azimuth angles.
            elevation (): Elevation angles.

        Returns:
            tuple: Cartesian coordinates.
        """
        assert (
            ndim(azimuth) == 1 and ndim(elevation) == 1
        ), "Inputs must be 1-dimensional"
        # elevation is π/2 - colatitude, so we calculate colatitude from elevation
        colatitude = pi / 2 - elevation
        x = sin(colatitude) * cos(azimuth)
        y = sin(colatitude) * sin(azimuth)
        z = cos(colatitude)
        return x, y, z

    @staticmethod
    def _cart_to_sph_colatitude(x, y, z) -> tuple:
        assert ndim(x) == 1 and ndim(y) == 1 and ndim(z)
        radius = 1
        azimuth = arctan2(y, x)
        azimuth = where(azimuth < 0, azimuth + 2 * pi, azimuth)
        colatitude = arccos(z / radius)
        return azimuth, colatitude

    @staticmethod
    def _cart_to_sph_elevation(x, y, z) -> tuple:
        assert ndim(x) == 1 and ndim(y) == 1 and ndim(z) == 1
        radius = 1
        azimuth = arctan2(y, x)
        azimuth = where(azimuth < 0, azimuth + 2 * pi, azimuth)
        elevation = pi / 2 - arccos(z / radius)  # elevation is π/2 - colatitude
        return azimuth, elevation