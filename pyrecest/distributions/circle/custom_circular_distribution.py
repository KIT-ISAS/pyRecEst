from collections.abc import Callable
from math import pi

from pyrecest.backend import array, mod

from ..abstract_custom_distribution import AbstractCustomDistribution
from .abstract_circular_distribution import AbstractCircularDistribution


class CustomCircularDistribution(
    AbstractCustomDistribution, AbstractCircularDistribution
):
    def __init__(self, f_: Callable, scale_by: float = 1, shift_by: float = 0):
        """
        Initializes a new instance of the CustomCircularDistribution class.

        Args:
            f_ (callable): The function to be used for the distribution.
            scale_by (float, optional): The scale factor for the distribution. Defaults to 1.
            shift_by (float, optional): The shift for the distribution. Defaults to 0.

        Note:
            It is the user's responsibility to ensure that f_ is a valid circular density,
            i.e., 2pi-periodic, nonnegative and normalized.
        """
        AbstractCircularDistribution.__init__(self)
        AbstractCustomDistribution.__init__(self, f_, scale_by)
        self.shift_by = shift_by

    def pdf(self, xs):
        """
        Computes the probability density function at xs.

        Args:
            xs (): The values at which to evaluate the pdf.

        Returns:
            : The value of the pdf at xs.
        """
        return AbstractCustomDistribution.pdf(self, mod(xs + self.shift_by, 2 * pi))

    def integrate(self, integration_boundaries=None) -> float:
        """
        Computes the integral of the pdf over the given boundaries.

        Args:
            integration_boundaries (, optional): The boundaries of the integral.
                Defaults to [0, 2 * pi].

        Returns:
            float: The value of the integral.
        """
        if integration_boundaries is None:
            integration_boundaries = array([0.0, 2.0 * pi])
        return AbstractCircularDistribution.integrate(self, integration_boundaries)
