import matplotlib.pyplot as plt
import numpy as np
from typing import Union

from beartype import beartype

from ..hypertorus.abstract_hypertoroidal_distribution import (
    AbstractHypertoroidalDistribution,
)
import numbers


class AbstractCircularDistribution(AbstractHypertoroidalDistribution):
    @beartype
    def __init__(self):
        AbstractHypertoroidalDistribution.__init__(self, dim=1)

    @beartype
    def cdf_numerical(self, xs: np.ndarray, starting_point: float = 0.0) -> np.ndarray:
        """
        Calculates the cumulative distribution function.

        Args:
            xs (np.ndarray): The 1D array to calculate the CDF on.
            starting_point (float, optional): Defaults to 0.

        Returns:
            np.ndarray: The computed CDF as a numpy array.
        """
        assert xs.ndim == 1, "xs must be a 1D array"

        return np.array([self._cdf_numerical_single(x, starting_point) for x in xs])

    @beartype
    def _cdf_numerical_single(self, x: Union[np.number, numbers.Real], starting_point: Union[np.number, numbers.Real]) -> Union[np.number, numbers.Real]:
        """Helper method for cdf_numerical"""
        starting_point_mod = np.mod(starting_point, 2 * np.pi)
        x_mod = np.mod(x, 2 * np.pi)

        if x_mod < starting_point_mod:
            return 1 - self.integrate_numerically([x_mod, starting_point_mod])

        return self.integrate_numerically([starting_point_mod, x_mod])

    def to_vm(self):
        """
        Convert to von Mises by trigonometric moment matching.

        Returns:
            vm (VMDistribution): Distribution with the same first trigonometric moment.
        """
        from .von_mises_distribution import VonMisesDistribution

        vm = VonMisesDistribution.from_moment(self.trigonometric_moment(1))
        return vm

    def to_wn(self):
        """
        Convert to wrapped normal by trigonometric moment matching.

        Returns:
            wn (WrappedNormalDistribution): Distribution with the same first trigonometric moment.
        """
        from .wrapped_normal_distribution import WrappedNormalDistribution

        wn = WrappedNormalDistribution.from_moment(self.trigonometric_moment(1))
        return wn

    @staticmethod
    def plot_circle(*args, **kwargs):
        theta = np.append(np.linspace(0, 2 * np.pi, 320), 0)
        p = plt.plot(np.cos(theta), np.sin(theta), *args, **kwargs)
        return p
