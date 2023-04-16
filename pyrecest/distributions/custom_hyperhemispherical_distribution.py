import numpy as np

from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution
from .custom_distribution import CustomDistribution


class CustomHyperhemisphericalDistribution(
    CustomDistribution, AbstractHyperhemisphericalDistribution
):
    def __init__(self, f, dim, shift_by: np.ndarray = None):
        CustomDistribution.__init__(self, f, dim)
        if shift_by is None:
            shift_by = np.zeros(self.dim + 1)
        assert shift_by.ndim == 1, "Shift_by must be a 1-D numpy vector."
        assert (
            shift_by.shape[0] == self.dim + 1
        ), "Shift_by vector length must match the dimension."
        self.shift_by = shift_by

    def pdf(self, xs):
        assert xs.shape[-1] == self.dim + 1
        # Reshape to properly handle the (d,) shape as well as the (n, d) case
        p = self.scale_by * self.f(
            np.reshape(xs, (-1, xs.shape[-1])) - self.shift_by[np.newaxis, :]
        )
        assert p.ndim <= 1, "Output format of pdf is not as expected"
        return p

    @staticmethod
    def from_distribution(dist):
        if isinstance(dist, AbstractHyperhemisphericalDistribution):
            return CustomHyperhemisphericalDistribution(
                lambda xs: dist.pdf(xs), dist.dim
            )
        elif isinstance(dist, BinghamDistribution):
            chhd = CustomHyperhemisphericalDistribution(
                lambda xs: dist.pdf(xs), dist.dim
            )
            chhd.scale_by = 2
            return
        elif isinstance(dist, AbstractHypersphericalDistribution):
            chhd_unnorm = CustomHyperhemisphericalDistribution(
                lambda xs: dist.pdf(xs), dist.dim
            )
            norm_const_inv = chhd_unnorm.integral()
            return CustomHyperhemisphericalDistribution(
                lambda xs: dist.pdf(xs) / norm_const_inv, dist.dim
            )
        else:
            raise ValueError("Input variable dist is of wrong class.")
