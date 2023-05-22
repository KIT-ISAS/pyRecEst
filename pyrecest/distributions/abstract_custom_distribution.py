import copy
import warnings
from abc import abstractmethod

from .abstract_distribution_type import AbstractDistributionType


class AbstractCustomDistribution(AbstractDistributionType):
    def __init__(self, f, scale_by=1):
        self.f = f
        self.scale_by = scale_by

    def pdf(self, xs):
        # Shifting is something for subclasses
        return self.scale_by * self.f(xs)

    @abstractmethod
    def integrate(self, integration_boundaries=None):
        pass

    def normalize(self, verify=None):
        cd = copy.deepcopy(self)

        integral = self.integrate()
        cd.scale_by = cd.scale_by / integral

        if verify and abs(cd.integrate()[0] - 1) > 0.001:
            warnings.warn("Density is not yet properly normalized.", UserWarning)

        return cd
