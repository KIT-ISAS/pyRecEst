from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution


class AbstractSphericalDistribution(
    AbstractSphereSubsetDistribution, AbstractHypersphericalDistribution
):
    def __init__(self):
        AbstractSphereSubsetDistribution.__init__(self)
        AbstractHypersphericalDistribution.__init__(self, dim=2)
