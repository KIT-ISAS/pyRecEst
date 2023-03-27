from abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution
from bingham_distribution import BinghamDistribution
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from custom_distribution import CustomDistribution

class CustomHyperhemisphericalDistribution(CustomDistribution, AbstractHyperhemisphericalDistribution):
    def __init__(self, f, dim):
        CustomDistribution.__init__(self, f, dim)
        
    @staticmethod
    def from_distribution(dist):
        if isinstance(dist, AbstractHyperhemisphericalDistribution):
            return CustomHyperhemisphericalDistribution(lambda xa: dist.pdf(xa), dist.dim)
        elif isinstance(dist, BinghamDistribution):
            chhd = CustomHyperhemisphericalDistribution(lambda xa: dist.pdf(xa), dist.dim)
            chhd.scale_by = 2
            return
        elif isinstance(dist, AbstractHypersphericalDistribution):
            chhd_unnorm = CustomHyperhemisphericalDistribution(lambda xa: dist.pdf(xa), dist.dim)
            norm_const_inv = chhd_unnorm.integral()
            return CustomHyperhemisphericalDistribution(lambda xa: dist.pdf(xa) / norm_const_inv, dist.dim)
        else:
            raise ValueError('Input variable dist is of wrong class.')