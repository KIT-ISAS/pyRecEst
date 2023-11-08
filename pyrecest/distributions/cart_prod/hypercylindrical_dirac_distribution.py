from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import cos, full, int32, int64, sin, sum, tile, stack, column_stack

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
        # Compute the cosine and sine components
        cos_vals = cos(self.d[:, :self.bound_dim])  # noqa: E203
        sin_vals = sin(self.d[:, :self.bound_dim])  # noqa: E203
        
        # Stack the cos, sin, and linear components along a new last dimension
        S = column_stack((cos_vals, sin_vals, self.d[:, self.bound_dim:]))  # noqa: E203
        
        # Perform the weighted sum using matrix multiplication
        return self.w @ S
