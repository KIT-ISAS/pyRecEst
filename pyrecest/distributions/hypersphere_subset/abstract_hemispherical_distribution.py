from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution


class AbstractHemisphericalDistribution(
    AbstractSphereSubsetDistribution, AbstractHyperhemisphericalDistribution
):
    def __init__(self):
        AbstractSphereSubsetDistribution.__init__(self)
        AbstractHyperhemisphericalDistribution.__init__(self, dim=2)
