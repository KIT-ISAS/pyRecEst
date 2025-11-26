from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hypersphere_subset_dirac_distribution import (
    AbstractHypersphereSubsetDiracDistribution,
)


class HyperhemisphericalDiracDistribution(
    AbstractHypersphereSubsetDiracDistribution, AbstractHyperhemisphericalDistribution
):
    def mean_axis(self):
        axis = super().mean_axis()
        if axis[-1] < 0:
            axis = -axis
        return axis
