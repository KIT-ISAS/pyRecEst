import numpy as np
from abc import ABC, abstractmethod
from abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution

class AbstractToroidalDistribution(AbstractHypertoroidalDistribution, ABC):
    def __init__(self):
        self.dim = 2
        super().__init__()

    @abstractmethod
    def pdf(self, xa):
        pass

    def integral(self, l=None, r=None):
        if l is None:
            l = np.array([0, 0])

        if r is None:
            r = np.array([2 * np.pi, 2 * np.pi])

        assert l.shape == (self.dim, )
        assert r.shape == (self.dim, )

        return self.integral_numerical(l, r)
