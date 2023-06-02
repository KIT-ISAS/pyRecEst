import warnings
from beartype import beartype
from .abstract_hemispherical_distribution import AbstractHemisphericalDistribution
from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution
from .custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution
from typing import Callable

class CustomHemisphericalDistribution(
    CustomHyperhemisphericalDistribution, AbstractHemisphericalDistribution
):
    @beartype
    def __init__(self, f: Callable):
        AbstractHemisphericalDistribution.__init__(self)
        CustomHyperhemisphericalDistribution.__init__(self, f, 2)

    @staticmethod
    @beartype
    def from_distribution(dist: "AbstractHypersphericalDistribution"):
        if dist.dim != 2:
            raise ValueError("Dimension of the distribution should be 2.")

        if isinstance(dist, AbstractHyperhemisphericalDistribution):
            return CustomHemisphericalDistribution(dist.pdf)
        if isinstance(dist, BinghamDistribution):
            chsd = CustomHemisphericalDistribution(dist.pdf)
            chsd.scale_by = 2
            return chsd
        if isinstance(dist, AbstractHypersphericalDistribution):
            warning_message = (
                "You are creating a CustomHyperhemispherical distribution based on a distribution on the full hypersphere. "
                + "Using numerical integration to calculate the normalization constant."
            )
            warnings.warn(warning_message, category=UserWarning)
            chhd_unnorm = CustomHyperhemisphericalDistribution(dist.pdf, dist.dim)
            norm_const_inv = chhd_unnorm.integrate()
            chsd = CustomHemisphericalDistribution(dist.pdf)
            chsd.scale_by = 1 / norm_const_inv
            return chsd

        raise ValueError("Input variable dist is of wrong class.")
