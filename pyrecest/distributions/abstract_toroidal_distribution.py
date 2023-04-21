from abc import ABC, abstractmethod

import numpy as np

from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class AbstractToroidalDistribution(AbstractHypertoroidalDistribution, ABC):
    def __init__(self):
        self.dim = 2
        super().__init__()

    @abstractmethod
    def pdf(self, xa):
        pass

    def integrate(self, left=None, right=None):
        left, right = self.prepare_integral_arguments(left, right)
        return self.integrate_numerically(left, right)

    def prepare_integral_arguments(self, left=None, right=None):
        if left is None:
            left = np.array([0, 0])

        if right is None:
            right = np.array([2 * np.pi, 2 * np.pi])

        assert left.shape == (self.dim,)
        assert right.shape == (self.dim,)

        return left, right
