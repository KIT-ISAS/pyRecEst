import numpy as np
from beartype import beartype

from ..abstract_uniform_distribution import AbstractUniformDistribution
from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHypersphereSubsetUniformDistribution(
    AbstractHypersphereSubsetDistribution, AbstractUniformDistribution
):
    """
    This is an abstract class for a uniform distribution over a subset of a hypersphere.
    """

    @beartype
    def pdf(self, xs: np.ndarray) -> np.ndarray:
        """
        Calculates the probability density function over the subset of the hypersphere.

        Args:
            xs (np.ndarray): Input data points.

        Returns:
            np.ndarray: Probability density at the given data points.
        """
        if xs.shape[-1] != self.input_dim:
            raise ValueError("Invalid shape of input data points.")
        manifold_size = self.get_manifold_size()
        if manifold_size == 0:
            raise ValueError("Manifold size cannot be zero.")
        if not isinstance(manifold_size, (int, float)):
            raise TypeError("Manifold size must be a numeric value.")
        p = (1 / manifold_size) * np.ones(xs.size // (self.dim + 1))
        return p
