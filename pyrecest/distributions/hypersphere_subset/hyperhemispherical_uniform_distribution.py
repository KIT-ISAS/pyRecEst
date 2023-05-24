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
    def sample(self, n: int):
        """
        Sample n points from the hyperhemispherical distribution.

        Args:
            n: number of points to sample.

        Returns:
            numpy.ndarray: n sampled points.
        """
        s = HypersphericalUniformDistribution(self.dim).sample(n)
        # Mirror ones with negative on the last dimension up for hemisphere. This
        # may give a disadvantage to ones with exactly zero at the first dimension but
        # since this is hit with quasi probability zero, we neglect
        # this.
        s = (1 - 2 * (s[-1, :] < 0)) * s
        return s

    def get_manifold_size(self):
        """
        Compute the size of the manifold.

        Returns:
            float: size of the manifold.
        """
        return (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.dim
            )
            / 2
        )
