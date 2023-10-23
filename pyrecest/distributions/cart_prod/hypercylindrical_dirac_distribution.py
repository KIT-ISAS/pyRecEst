from typing import Union

from pyrecest.backend import cos, full, int32, int64, sin, sum, tile

from ..hypertorus.hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution
from .lin_bounded_cart_prod_dirac_distribution import (
    LinBoundedCartProdDiracDistribution,
)


class HypercylindricalDiracDistribution(
    LinBoundedCartProdDiracDistribution, AbstractHypercylindricalDistribution
):
    def __init__(self, bound_dim: Union[int, int32, int64], d, w=None):
        AbstractHypercylindricalDistribution.__init__(
            self, bound_dim, d.shape[-1] - bound_dim
        )
        LinBoundedCartProdDiracDistribution.__init__(self, d=d, w=w)

    def pdf(self, xs):
        return LinBoundedCartProdDiracDistribution.pdf(self, xs)

    def marginalize_periodic(self):
        return LinBoundedCartProdDiracDistribution.marginalize_periodic(self)

    def marginalize_linear(self):
        return HypertoroidalDiracDistribution(
            self.d[:, 0 : self.bound_dim], self.w  # noqa: E203
        )

    def hybrid_moment(self):
        # Specific for Cartesian products of hypertori and R^lin_dim
        S = full((self.bound_dim * 2 + self.lin_dim, self.d.shape[0]), float("NaN"))
        S[2 * self.bound_dim :, :] = self.d[:, self.bound_dim :].T  # noqa: E203

        for i in range(self.bound_dim):
            S[2 * i, :] = cos(self.d[:, i])  # noqa: E203
            S[2 * i + 1, :] = sin(self.d[:, i])  # noqa: E203

        return sum(tile(self.w, (self.lin_dim + 2 * self.bound_dim, 1)) * S, axis=1)
