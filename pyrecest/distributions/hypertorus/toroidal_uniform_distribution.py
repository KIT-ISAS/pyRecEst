import copy

from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .hypertoroidal_uniform_distribution import HypertoroidalUniformDistribution


class ToroidalUniformDistribution(
    HypertoroidalUniformDistribution, AbstractToroidalDistribution
):
    def get_manifold_size(self):
        return AbstractToroidalDistribution.get_manifold_size(self)

    def shift(self, _):
        return copy.deepcopy(self)
