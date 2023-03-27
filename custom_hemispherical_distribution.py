from custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution
from abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution
from bingham_distribution import BinghamDistribution
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
import warnings

class CustomHemisphericalDistribution(CustomHyperhemisphericalDistribution):
    def __init__(self, f):
        CustomHyperhemisphericalDistribution.__init__(self, f, 3)

    @staticmethod
    def from_distribution(dist):
        if dist.dim != 3:
            raise ValueError("Dimension of the distribution should be 3.")

        if isinstance(dist, AbstractHyperhemisphericalDistribution):
            return CustomHemisphericalDistribution(lambda xa: dist.pdf(xa))
        elif isinstance(dist, BinghamDistribution):
            chsd = CustomHemisphericalDistribution(lambda xa: dist.pdf(xa))
            chsd.scale_by = 2
            return chsd
        elif isinstance(dist, AbstractHypersphericalDistribution):
            warning_message = 'You are creating a CustomHyperhemispherical distribution based on a distribution on the full hypersphere. ' + \
                  'Using numerical integration to calculate the normalization constant.'
            warnings.warn(warning_message, category=UserWarning)
            chhd_unnorm = CustomHyperhemisphericalDistribution(lambda xa: dist.pdf(xa), dist.dim)
            norm_const_inv = chhd_unnorm.integral()
            chsd = CustomHemisphericalDistribution(lambda xa: dist.pdf(xa))
            chsd.scale_by = 1 / norm_const_inv
            return chsd
        else:
            raise ValueError("Input variable dist is of wrong class.")