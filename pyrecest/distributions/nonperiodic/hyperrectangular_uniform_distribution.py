from ..abstract_uniform_distribution import AbstractUniformDistribution
from .abstract_hyperrectangular_distribution import AbstractHyperrectangularDistribution


class HyperrectangularUniformDistribution(
    AbstractUniformDistribution, AbstractHyperrectangularDistribution
):
    def __init__(self, bounds):
        AbstractUniformDistribution.__init__(self)
        AbstractHyperrectangularDistribution.__init__(self, bounds)

    def get_manifold_size(self):
        return AbstractHyperrectangularDistribution.get_manifold_size(self)