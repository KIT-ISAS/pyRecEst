from .nonperiodic.abstract_hyperrectangular_distribution import AbstractHyperrectangularDistribution
from .custom_distribution import CustomDistribution

class CustomHyperrectangularDistribution(AbstractHyperrectangularDistribution, CustomDistribution):
    def __init__(self, f, bounds):
        AbstractHyperrectangularDistribution.__init__(self, bounds)
        CustomDistribution.__init__(self, f, self.dim)
