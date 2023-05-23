from abc import abstractmethod

import numpy as np

from .abstract_distribution_type import AbstractDistributionType


class AbstractUniformDistribution(AbstractDistributionType):
    # Uniform distribution.
    def pdf(self, xs):
        return 1 / self.get_manifold_size() * np.ones(xs.shape[0])

    @abstractmethod
    def get_manifold_size(self):
        pass

    def mode(self):
        raise NotImplementedError("Mode not available for uniform distribution")
