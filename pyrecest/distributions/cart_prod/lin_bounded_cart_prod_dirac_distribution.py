from abc import abstractmethod

import numpy as np

from ..abstract_dirac_distribution import AbstractDiracDistribution
from ..nonperiodic.linear_dirac_distribution import LinearDiracDistribution
from .abstract_lin_bounded_cart_prod_distribution import (
    AbstractLinBoundedCartProdDistribution,
)


class LinBoundedCartProdDiracDistribution(
    AbstractDiracDistribution, AbstractLinBoundedCartProdDistribution
):
    def __init__(self, bound_dim, d, w=None):
        AbstractLinBoundedCartProdDistribution.__init__(
            self, bound_dim=bound_dim, lin_dim=d.shape[1] - bound_dim
        )
        AbstractDiracDistribution.__init__(self, d, w)

    def marginalize_periodic(self):
        return LinearDiracDistribution(
            self.d[:, self.bound_dim :], self.w # noqa: E203
        )

    def linear_mean(self):
        return self.marginalize_periodic().mean()

    def linear_covariance(self):
        return self.marginalize_periodic().covariance()

    @abstractmethod
    def marginalize_linear(self):
        pass

    def hybrid_mean(self):
        periodic = self.marginalize_linear()
        linear = self.marginalize_periodic()

        return np.concatenate((periodic.mean_direction(), linear.mean()))

    @classmethod
    def from_distribution(cls, distribution, n_samples):
        """
        Needs to be overwritten to allow the specification of bound_dim for Cartesian
        products of bounded and Euclidean manifolds
        """
        assert cls.is_valid_for_conversion(distribution)
        samples = distribution.sample(n_samples)
        return cls(distribution.bound_dim, samples)
