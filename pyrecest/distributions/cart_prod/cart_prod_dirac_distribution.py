from ..hypersphere_subset.abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from ..hypersphere_subset.abstract_hypersphere_subset_dirac_distribution import (
    AbstractHypersphereSubsetDiracDistribution,
)
from ..hypersphere_subset.abstract_hyperspherical_distribution import (
    AbstractHypersphericalDistribution,
)


class HyperhemisphericalDiracDistribution(
    AbstractHypersphereSubsetDiracDistribution, AbstractHyperhemisphericalDistribution
):
    pass


class HypersphericalDiracDistribution(
    AbstractHypersphereSubsetDiracDistribution, AbstractHypersphericalDistribution
):
    pass
