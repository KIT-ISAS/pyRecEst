from math import pi
from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import cos, empty, int32, int64, linalg, random, sin, sqrt

from .abstract_hypersphere_subset_uniform_distribution import (
    AbstractHypersphereSubsetUniformDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class HypersphericalUniformDistribution(
    AbstractHypersphericalDistribution, AbstractHypersphereSubsetUniformDistribution
):
    def __init__(self, dim: Union[int, int32, int64]):
        AbstractHypersphereSubsetUniformDistribution.__init__(self, dim)

    def pdf(self, xs):
        return AbstractHypersphereSubsetUniformDistribution.pdf(self, xs)

    def sample(self, n: Union[int, int32, int64]):
        assert isinstance(n, int) and n > 0, "n must be a positive integer"

        if self.dim == 2:
            s = empty(
                (
                    n,
                    self.dim + 1,
                )
            )
            phi = 2 * pi * random.rand(n)
            s[:, 2] = random.rand(n) * 2.0 - 1.0
            r = sqrt(1 - s[:, 2] ** 2)
            s[:, 0] = r * cos(phi)
            s[:, 1] = r * sin(phi)
        else:
            samples_unnorm = random.normal(0.0, 1.0, (n, self.dim + 1))
            s = samples_unnorm / linalg.norm(samples_unnorm, axis=1).reshape(-1, 1)
        return s

    def get_manifold_size(self):
        return AbstractHypersphericalDistribution.get_manifold_size(self)
