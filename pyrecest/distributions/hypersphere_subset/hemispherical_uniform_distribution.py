from .abstract_hemispherical_distribution import AbstractHemisphericalDistribution
from .abstract_hypersphere_subset_uniform_distribution import (
    AbstractHypersphereSubsetUniformDistribution,
)


class HemisphericalUniformDistribution(
    AbstractHypersphereSubsetUniformDistribution, AbstractHemisphericalDistribution
):
    pass