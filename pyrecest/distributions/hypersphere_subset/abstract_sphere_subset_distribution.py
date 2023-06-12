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
        assert phi.ndim == 1 and theta.ndim == 1, "Input must be 1-dimensional"
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
        # Assuming radius is 1
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
        # Assuming radius is 1
        x = np.cos(elevation) * np.cos(azimuth)
        y = np.cos(elevation) * np.sin(azimuth)
        z = np.sin(elevation)
        return x, y, z

    @staticmethod
    @beartype
    def _cart_to_sph_colatitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        radius = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
        colatitude = np.arccos(z / radius)
        return azimuth, colatitude

    @staticmethod
    @beartype
    def _cart_to_sph_elevation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        radius = np.sqrt(x**2 + y**2 + z**2)
        azimuth = np.arctan2(y, x)
        azimuth = np.where(azimuth < 0, azimuth + 2 * np.pi, azimuth)
        elevation = np.arccos(z / radius)
        return azimuth, elevation
