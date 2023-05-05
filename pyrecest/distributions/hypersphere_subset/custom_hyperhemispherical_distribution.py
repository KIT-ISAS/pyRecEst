import numpy as np

from ..custom_distribution import CustomDistribution
from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution


class CustomHyperhemisphericalDistribution(
    CustomDistribution, AbstractHyperhemisphericalDistribution
):

    def pdf(self, xs):
        assert xs.shape[-1] == self.dim + 1
        # Reshape to properly handle the (d,) shape as well as the (n, d) case
        p = self.scale_by * self.f(
            np.reshape(xs, (-1, xs.shape[-1]))
        )
        assert p.ndim <= 1, "Output format of pdf is not as expected"
        return p

    @staticmethod
    def from_distribution(dist):
        if isinstance(dist, AbstractHyperhemisphericalDistribution):
            return CustomHyperhemisphericalDistribution(dist.pdf, dist.dim)
        
        if isinstance(dist, BinghamDistribution):
            chhd = CustomHyperhemisphericalDistribution(dist.pdf, dist.dim)
            chhd.scale_by = 2
            return chhd
        
        if isinstance(dist, AbstractHypersphericalDistribution):
            chhd_unnorm = CustomHyperhemisphericalDistribution(dist.pdf, dist.dim)
            norm_const_inv = chhd_unnorm.integrate()
            return CustomHyperhemisphericalDistribution(
                dist.pdf / norm_const_inv, dist.dim
            )
        
        raise ValueError("Input variable dist is of wrong class.")
