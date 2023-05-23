import numpy as np

from ..abstract_custom_distribution import AbstractCustomDistribution
from .abstract_circular_distribution import AbstractCircularDistribution


class CustomCircularDistribution(
    AbstractCustomDistribution, AbstractCircularDistribution
):
    def __init__(self, f_, scale_by=1, shift_by=0):
        """
        It is the user's responsibility to ensure that f is a valid
        circular density, i.e., 2pi-periodic, nonnegative and
        normalized.
        """
        AbstractCircularDistribution.__init__(self)
        AbstractCustomDistribution.__init__(self, f_, scale_by)
        self.shift_by = shift_by

    def pdf(self, xs):
        super().pdf(np.mod(xs + self.shift_by, 2 * np.pi))

    def integrate(self, integration_boundaries=np.array([0, 2 * np.pi])):
        return AbstractCircularDistribution.integrate(self, integration_boundaries)
