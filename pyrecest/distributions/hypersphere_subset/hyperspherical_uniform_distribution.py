import numpy as np
from beartype import beartype

from .abstract_hypersphere_subset_uniform_distribution import (
    AbstractHypersphereSubsetUniformDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class HypersphericalUniformDistribution(
    AbstractHypersphericalDistribution, AbstractHypersphereSubsetUniformDistribution
):
    @beartype
    def __init__(self, dim: int | np.int32 | np.int64):
        AbstractHypersphereSubsetUniformDistribution.__init__(self, dim)

    @beartype
    def pdf(self, xs: np.ndarray):
        return AbstractHypersphereSubsetUniformDistribution.pdf(self, xs)

    @beartype
    def sample(self, n: int | np.int32 | np.int64):
        assert isinstance(n, int) and n > 0, "n must be a positive integer"

        if self.dim == 2:
            s = np.empty(
                (
                    n,
                    self.dim + 1,
                )
            )
            phi = 2 * np.pi * np.random.rand(n)
            s[:, 2] = np.random.rand(n) * 2 - 1
            r = np.sqrt(1 - s[:, 2] ** 2)
            s[:, 0] = r * np.cos(phi)
            s[:, 1] = r * np.sin(phi)
        else:
            samples_unnorm = np.random.randn(n, self.dim + 1)
            s = samples_unnorm / np.linalg.norm(samples_unnorm, axis=1, keepdims=True)
        return s

    def get_manifold_size(self):
        return AbstractHypersphericalDistribution.get_manifold_size(self)
