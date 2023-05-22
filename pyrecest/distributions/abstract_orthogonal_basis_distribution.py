import copy
from abc import abstractmethod

import numpy as np

from .abstract_distribution_type import AbstractDistributionType


class AbstractOrthogonalBasisDistribution(AbstractDistributionType):
    def __init__(self, coeff_mat, transformation):
        self.transformation = transformation
        self.coeff_mat = coeff_mat
        self.normalize_in_place()

    @abstractmethod
    def normalize_in_place(self):
        pass

    @abstractmethod
    def value(self, xa):
        pass

    def normalize(self):
        result = copy.deepcopy(self)
        return result.normalize_in_place()

    def pdf(self, xa):
        val = self.value(xa)
        if self.transformation == "sqrt":
            assert all(np.imag(val) < 0.0001)
            return np.real(val) ** 2

        if self.transformation == "identity":
            return val

        if self.transformation == "log":
            print("Warning: Density may not be normalized")
            return np.exp(val)

        raise ValueError("Transformation not recognized or unsupported")
