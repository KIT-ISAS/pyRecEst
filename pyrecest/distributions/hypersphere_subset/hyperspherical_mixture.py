from ..abstract_mixture import AbstractMixture
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class HypersphericalMixture(AbstractMixture, AbstractHypersphericalDistribution):
    def __init__(self, dists, w):
        AbstractHypersphericalDistribution.__init__(self, dim=dists[0].dim)
        assert all(
            isinstance(dist, AbstractHypersphericalDistribution) for dist in dists
        ), "dists must be a list of hyperspherical distributions"

        AbstractMixture.__init__(self, dists, w)
