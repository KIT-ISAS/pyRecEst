from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hypersphere_subset_dirac_distribution import (
    AbstractHypersphereSubsetDiracDistribution,
)


class HyperhemisphericalDiracDistribution(
    AbstractHypersphereSubsetDiracDistribution, AbstractHyperhemisphericalDistribution
):
    pass