import numpy as np

from ..abstract_uniform_distribution import AbstractUniformDistribution
from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHypersphereSubsetUniformDistribution(
    AbstractHypersphereSubsetDistribution, AbstractUniformDistribution
):
    def pdf(self, xs):
        p = (1 / self.get_manifold_size()) * np.ones(xs.size // (self.dim + 1))
        return p
