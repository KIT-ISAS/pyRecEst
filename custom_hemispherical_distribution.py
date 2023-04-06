import warnings
import numpy as np

from custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution
from abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution
from bingham_distribution import BinghamDistribution
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class CustomHemisphericalDistribution(CustomHyperhemisphericalDistribution):
    def __init__(self, f):
        CustomHyperhemisphericalDistribution.__init__(self, f, 2)
    
    @staticmethod
    def from_distribution(dist):
        if dist.dim != 2:
            raise ValueError("Dimension of the distribution should be 2.")

        if isinstance(dist, AbstractHyperhemisphericalDistribution):
            return CustomHemisphericalDistribution(lambda xs: dist.pdf(xs))
        elif isinstance(dist, BinghamDistribution):
            chsd = CustomHemisphericalDistribution(lambda xs: dist.pdf(xs))
            chsd.scale_by = 2
            return chsd
        elif isinstance(dist, AbstractHypersphericalDistribution):
            warning_message = 'You are creating a CustomHyperhemispherical distribution based on a distribution on the full hypersphere. ' + \
                  'Using numerical integration to calculate the normalization constant.'
            warnings.warn(warning_message, category=UserWarning)
            chhd_unnorm = CustomHyperhemisphericalDistribution(lambda xs: dist.pdf(xs), dist.dim)
            norm_const_inv = chhd_unnorm.integral()
            chsd = CustomHemisphericalDistribution(lambda xs: dist.pdf(xs))
            chsd.scale_by = 1 / norm_const_inv
            return chsd
        else:
            raise ValueError("Input variable dist is of wrong class.")