from collections.abc import Callable

import numpy as np
from beartype import beartype

from ..abstract_custom_distribution import AbstractCustomDistribution
from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution


class CustomHyperhemisphericalDistribution(
    AbstractCustomDistribution, AbstractHyperhemisphericalDistribution
):
    @beartype
    def __init__(
        self, f: Callable, dim: int | np.int32 | np.int64, scale_by: float = 1
    ):
        """
        Initialize a CustomHyperhemisphericalDistribution.

        :param f: function to be used for custom distribution.
        :param dim: dimension of the distribution.
        :param scale_by: scaling factor for the distribution, default is 1.
        """
        AbstractHyperhemisphericalDistribution.__init__(self, dim=dim)
        AbstractCustomDistribution.__init__(self, f=f, scale_by=scale_by)

    @beartype
    def pdf(self, xs):
        """
        Calculate the probability density function at given points.

        :param xs: points where the pdf will be calculated.
        :return: numpy.ndarray: pdf values at given points.
        """
        assert xs.shape[-1] == self.dim + 1
        p = self.scale_by * self.f(xs)
        assert p.ndim <= 1, "Output format of pdf is not as expected"
        return p

    def integrate(self, integration_boundaries=None):
        """
        Integrate the pdf over given boundaries.

        :param integration_boundaries: boundaries of the integration, if None, the integration will be performed over the whole domain.
        :return: float: result of the integration.
        """
        return AbstractHyperhemisphericalDistribution.integrate(
            self, integration_boundaries
        )

    @staticmethod
    @beartype
    def from_distribution(dist: "AbstractHypersphericalDistribution"):
        """
        Create a CustomHyperhemisphericalDistribution from another distribution.

        :param dist: the distribution from which the CustomHyperhemisphericalDistribution will be created.
        :return: CustomHyperhemisphericalDistribution: the resulting distribution.
        :raises ValueError: if the type of dist is not supported.
        """
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
