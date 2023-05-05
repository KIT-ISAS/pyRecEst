from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)


class AbstractHemisphericalDistribution(AbstractHyperhemisphericalDistribution):
    def __init__(self):
        AbstractHyperhemisphericalDistribution.__init__(self, dim=2)
