from .abstract_toroidal_distribution import AbstractToroidalDistribution
from .custom_hypertoroidal_distribution import CustomHypertoroidalDistribution


class CustomToroidalDistribution(
    CustomHypertoroidalDistribution, AbstractToroidalDistribution
):
    def __init__(self, f):
        AbstractToroidalDistribution.__init__(self)
        CustomHypertoroidalDistribution.__init__(self, f, 2)
