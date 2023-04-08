from abc import abstractmethod
from .abstract_distribution import AbstractDistribution

class AbstractUniformDistribution(AbstractDistribution):
    # Uniform distribution.

    @abstractmethod
    def get_manifold_size(self):
        pass

    def mode(self):
        raise NotImplementedError('Mode not available for uniform distribution')
