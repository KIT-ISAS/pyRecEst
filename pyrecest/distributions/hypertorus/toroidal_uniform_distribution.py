from .hypertoroidal_uniform_distribution import HypertoroidalUniformDistribution
from .abstract_toroidal_distribution import AbstractToroidalDistribution

class ToroidalUniformDistribution(HypertoroidalUniformDistribution, AbstractToroidalDistribution):
    def get_manifold_size(self):
        return AbstractToroidalDistribution.get_manifold_size(self)