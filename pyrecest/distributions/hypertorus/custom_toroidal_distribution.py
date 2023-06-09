from .custom_hypertoroidal_distribution import CustomHypertoroidalDistribution
from .abstract_toroidal_distribution import AbstractToroidalDistribution

class CustomToroidalDistribution(CustomHypertoroidalDistribution, AbstractToroidalDistribution):
    def __init__(self, f):
        AbstractToroidalDistribution.__init__(self)
        CustomHypertoroidalDistribution.__init__(self, f, 2)