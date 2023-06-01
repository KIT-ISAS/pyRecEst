from abc import abstractmethod

from .abstract_bounded_domain_distribution import AbstractBoundedDomainDistribution
from beartype import beartype
import numpy as np
from typing import Union

class AbstractPeriodicDistribution(AbstractBoundedDomainDistribution):
    """Abstract class for a distributions on periodic manifolds."""

    @beartype
    def __init__(self, dim: Union[int, np.int32, np.int64]):
        super().__init__(dim=dim)

    @beartype
    def mean(self) -> np.ndarray:
        """
        Convenient access to mean_direction to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype: np.ndarray
        """
        return self.mean_direction()

    @abstractmethod
    def mean_direction(self) -> np.ndarray:
        """
        Abstract method to compute the mean direction of the distribution.

        Returns
        -------
        mean_direction: np.ndarray
            The mean direction of the distribution.
        """
