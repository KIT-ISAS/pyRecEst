from pyrecest.backend import ones
from abc import abstractmethod

import numpy as np
from beartype import beartype

from .abstract_distribution_type import AbstractDistributionType


class AbstractUniformDistribution(AbstractDistributionType):
    """Abstract class for a uniform distribution on a manifold."""

    def pdf(self, xs: np.ndarray) -> np.ndarray:
        """Compute the probability density function at each point in xs.

        :param xs: Points at which to compute the pdf.
        :type xs: np.ndarray

        :return: The pdf evaluated at each point in xs.
        :rtype: np.ndarray
        """
        return 1 / self.get_manifold_size() * ones(xs.shape[0])

    @abstractmethod
    def get_manifold_size(self) -> np.ndarray:
        """
        Compute the probability density function at each point in xs.

        :param xs: Points at which to compute the pdf.
        :type xs: np.ndarray

        :return: The pdf evaluated at each point in xs.
        :rtype: np.ndarray
        """

    def mode(self):
        """Mode is not defined for a uniform distribution."""
        raise AttributeError("Mode not available for uniform distribution")