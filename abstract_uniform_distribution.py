from abc import abstractmethod
from abstract_distribution import AbstractDistribution
import numpy as np

class AbstractUniformDistribution(AbstractDistribution):
    # Uniform distribution.

    def pdf(self, xa):
        p = (1 / self.get_manifold_size()) * np.ones(xa.size//self.dim)
        return p

    @abstractmethod
    def get_manifold_size(self):
        pass

    def mode(self):
        raise NotImplementedError('Mode not available for uniform distribution')
