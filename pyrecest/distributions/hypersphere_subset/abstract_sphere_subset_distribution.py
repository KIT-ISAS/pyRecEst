import numpy as np
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
    @beartype
    def sph_to_cart(phi: np.ndarray, theta: np.ndarray, mode="colatitude") -> tuple:
        """
        Convert spherical coordinates to Cartesian coordinates.

        Args:
            phi (np.ndarray): Azimuth angles.
            theta (np.ndarray): Colatitude angles or elevation angles based on the mode.
            mode (str): Either 'colatitude' or 'elevation'.

        Returns:
            tuple: Cartesian coordinates.
        """
        assert np.ndim(phi) == 1 and np.ndim(theta) == 1, "Inputs must be 1-dimensional"
        if mode == "colatitude":
            x, y, z = AbstractSphereSubsetDistribution._sph_to_cart_colatitude(
                phi, theta
            )
        elif mode == "elevation":
            x, y, z = AbstractSphereSubsetDistribution._sph_to_cart_elevation(
                phi, theta
            )
        else:
            raise ValueError("mode must be either 'colatitude' or 'elevation'")

        return x, y, z

    @staticmethod
    @beartype
    def cart_to_sph(
        x: np.ndarray, y: np.ndarray, z: np.ndarray, mode="colatitude"
    ) -> tuple:
        """
        Convert Cartesian coordinates to spherical coordinates.

        Args:
            x (np.ndarray): X coordinates.
            y (np.ndarray): Y coordinates.
            z (np.ndarray): Z coordinates.

        Returns:
            tuple: Spherical coordinates.
        """
        assert (
            np.ndim(x) == 1 and np.ndim(y) == 1 and np.ndim(z)
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
            raise ValueError("mode must be either 'colatitude' or 'elevation'")

        return phi, theta

    @staticmethod
    @beartype
    def _sph_to_cart_colatitude(azimuth: np.ndarray, colatitude: np.ndarray) -> tuple:
        assert np.ndim(azimuth) == 1 and np.ndim(
            colatitude
        ), "Inputs must be 1-dimensional"
        x = np.sin(colatitude) * np.cos(azimuth)
        y = np.sin(colatitude) * np.sin(azimuth)
        z = np.cos(colatitude)
        return x, y, z

    @staticmethod
    @beartype
    def _sph_to_cart_elevation(azimuth: np.ndarray, elevation: np.ndarray) -> tuple:
        """
        Convert spherical coordinates (using elevation) to Cartesian coordinates.
        Assumes a radius of 1.

        Args:
            azimuth (np.ndarray): Azimuth angles.
            elevation (np.ndarray): Elevation angles.

        Returns:
            tuple: Cartesian coordinates.
        """
        assert (
            np.ndim(azimuth) == 1 and np.ndim(elevation) == 1
        ), "Inputs must be 1-dimensional"
        # elevation is π/2 - colatitude, so we calculate colatitude from elevation
        colatitude = np.pi / 2 - elevation
        x = np.sin(colatitude) * np.cos(azimuth)
        y = np.sin(colatitude) * np.sin(azimuth)
        z = np.cos(colatitude)
        return x, y, z

    @staticmethod
    @beartype
    def _cart_to_sph_colatitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        assert np.ndim(x) == 1 and np.ndim(y) == 1 and np.ndim(z)
        radius = 1
        azimuth = np.arctan2(y, x)
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
        colatitude = np.arccos(z / radius)
        return azimuth, colatitude

    @staticmethod
    @beartype
    def _cart_to_sph_elevation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        assert np.ndim(x) == 1 and np.ndim(y) == 1 and np.ndim(z) == 1
        radius = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
        elevation = np.pi / 2 - np.arccos(z / radius)  # elevation is π/2 - colatitude
        return azimuth, elevation
