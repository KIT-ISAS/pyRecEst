from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution


class AbstractHemisphericalDistribution(
    AbstractSphereSubsetDistribution, AbstractHyperhemisphericalDistribution
):
    """
    This abstract class represents a distribution over the hemisphere.
    It inherits from both AbstractSphereSubsetDistribution and AbstractHyperhemisphericalDistribution.
    The class is meant to be subclassed and should not be instantiated directly.
    """

    def __init__(self):
        """
        Initializes a new instance of the AbstractHemisphericalDistribution class.
        """
        AbstractSphereSubsetDistribution.__init__(self)
        AbstractHyperhemisphericalDistribution.__init__(self, dim=2)