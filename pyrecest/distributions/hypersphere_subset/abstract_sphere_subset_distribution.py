import numpy as np
from beartype import beartype
from scipy.spatial.transform import Rotation as R

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
    def sph_to_cart(azimuth: np.ndarray, elevation: np.ndarray) -> tuple:
        """
        Convert spherical coordinates to Cartesian coordinates.

        Args:
            azimuth (np.ndarray): Azimuth angles.
            elevation (np.ndarray): Elevation angles.

        Returns:
            tuple: Cartesian coordinates.
        """
        assert azimuth.ndim == 1 and elevation.ndim == 1, "Input must be 1-dimensional"
        x, y, z = (
            R.from_euler(
                "ZYX", np.column_stack((azimuth, elevation, np.zeros(np.size(azimuth))))
            )
            .apply(np.array([1, 0, 0]).reshape(1, 3))
            .T
        )
        return x, y, -z

    @staticmethod
    @beartype
    def cart_to_sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        """
        Convert Cartesian coordinates to spherical coordinates.

        Args:
            x (np.ndarray): X coordinates.
            y (np.ndarray): Y coordinates.
            z (np.ndarray): Z coordinates.

        Returns:
            tuple: Spherical coordinates.
        """
        return AbstractSphereSubsetDistribution._cart_to_sph_colatitude(x, y, z)

    @staticmethod
    @beartype
    def _cart_to_sph_colatitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        hxy = np.hypot(x, y)
        colatitude = np.arctan2(hxy, z)
        azimuth = np.arctan2(y, x)
        return azimuth, colatitude

    @staticmethod
    @beartype
    def _cart_to_sph_elevation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple:
        hxy = np.hypot(x, y)
        elevation = np.arctan2(z, hxy)
        azimuth = np.arctan2(y, x)
        return azimuth, elevation
