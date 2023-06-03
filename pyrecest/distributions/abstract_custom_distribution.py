import copy
import warnings
from abc import abstractmethod
from collections.abc import Callable

import numpy as np
from beartype import beartype

from .abstract_distribution_type import AbstractDistributionType


class AbstractCustomDistribution(AbstractDistributionType):
    """
    Abstract class for creating distributions based on callable functions.

    This class accepts a function `f` that calculates the probability density function
    and a scaling factor `scale_by` to adjust the PDF.

    Methods:
    - pdf(xs : Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        Compute the probability density function at given points.
    - integrate(integration_boundaries: Optional[Union[float, Tuple[float, float]]] = None) -> float:
        Calculate the integral of the probability density function.
    - normalize(verify : Optional[bool] = None) -> AbstractCustomDistribution:
        Normalize the PDF such that its integral is 1. Returns a copy of the original distribution.
    """

    @beartype
    def __init__(self, f: Callable[[np.ndarray], np.ndarray], scale_by=1):
        """
        Initialize AbstractCustomDistribution.

        :param f: The function that calculates the probability density function.
        :param scale_by: Scaling factor to adjust the PDF, default is 1.
        """
        self.f = f
        self.scale_by = scale_by

    @beartype
    def pdf(self, xs: np.ndarray) -> np.ndarray:
        """
        Compute the probability density function at given points.

        :param xs: Points at which to compute the PDF.
        :returns: PDF values at given points.
        """
        # Shifting is something for subclasses
        return self.scale_by * self.f(xs)

    @abstractmethod
    @beartype
    def integrate(self, integration_boundaries=None):
        """
        Calculate the integral of the probability density function.

        :param integration_boundaries: The boundaries of integration, default is None.
        :returns: The integral of the PDF.
        """

    @beartype
    def normalize(self, verify: bool | None = None) -> "AbstractCustomDistribution":
        """
        Normalize the PDF such that its integral is 1.

        :param verify: Whether to verify if the density is properly normalized, default is None.
        :returns: A copy of the original distribution, with the PDF normalized.
        """
        cd = copy.deepcopy(self)

        integral = self.integrate()
        cd.scale_by = cd.scale_by / integral

        if verify and abs(cd.integrate()[0] - 1) > 0.001:
            warnings.warn("Density is not yet properly normalized.", UserWarning)

        return cd
