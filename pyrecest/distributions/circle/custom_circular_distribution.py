from .abstract_circular_distribution import AbstractCircularDistribution
from ..custom_distribution import CustomDistribution
import numpy as np

class CustomCircularDistribution(CustomDistribution, AbstractCircularDistribution):
    def __init__(self, f_):
        """
        It is the user's responsibility to ensure that f is a valid
        circular density, i.e., 2pi-periodic, nonnegative and
        normalized.
        """
        AbstractCircularDistribution.__init__(self)
        CustomDistribution.__init__(self, f_, 1)
        
    def integrate(self, integration_boundaries=np.array([0, 2*np.pi])):
        return AbstractCircularDistribution.integrate(self, integration_boundaries)
