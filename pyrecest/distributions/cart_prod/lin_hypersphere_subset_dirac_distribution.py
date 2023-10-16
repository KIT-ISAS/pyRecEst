from pyrecest.backend import int64
from pyrecest.backend import int32
import numpy as np

from .abstract_lin_hyperhemisphere_cart_prod_distribution import (
    AbstractLinHypersphereSubsetCartProdDistribution,
)
from .lin_bounded_cart_prod_dirac_distribution import (
    LinBoundedCartProdDiracDistribution,
)


class LinHypersphereSubsetCartProdDiracDistribution(
    LinBoundedCartProdDiracDistribution,
    AbstractLinHypersphereSubsetCartProdDistribution,
):
    def __init__(self, bound_dim: int | int32 | int64, d, w=None):
        AbstractLinHypersphereSubsetCartProdDistribution.__init__(
            self, bound_dim, d.shape[-1] - bound_dim - 1
        )
        LinBoundedCartProdDiracDistribution.__init__(self, d=d, w=w)
