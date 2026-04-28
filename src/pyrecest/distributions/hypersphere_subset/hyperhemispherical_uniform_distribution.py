from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import int32, int64

from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)
from .abstract_hypersphere_subset_uniform_distribution import (
    AbstractHypersphereSubsetUniformDistribution,
)
from .hyperspherical_uniform_distribution import HypersphericalUniformDistribution


class HyperhemisphericalUniformDistribution(
    AbstractHyperhemisphericalDistribution, AbstractHypersphereSubsetUniformDistribution
):
    def sample(self, n: Union[int, int32, int64]):
        """
        Sample n points from the hyperhemispherical distribution.

        Args:
            n: number of points to sample.

        Returns:
            numpy.ndarray: n sampled points.
        """
        s = HypersphericalUniformDistribution(self.dim).sample(n)
        # Mirror samples with negative last coordinate up to the hemisphere.
        # Samples exactly on the equator are left unchanged; this has probability zero.
        s = (1 - 2 * (s[:, -1:] < 0)) * s
        return s

    def get_manifold_size(self) -> float:
        """
        Compute the size of the manifold.

        Returns:
            float: size of the manifold.
        """
        return (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.dim
            )
            / 2.0
        )
