from pyrecest.backend import abs
import numpy as np

from ..abstract_se3_distribution import AbstractSE3Distribution
from .lin_bounded_cart_prod_dirac_distribution import (
    LinBoundedCartProdDiracDistribution,
)


class LinHypersphereCartProdDiracDistribution(
    LinBoundedCartProdDiracDistribution, AbstractSE3Distribution
):
    def __init__(self, bound_dim, d, w=None):
        assert (
            np.max(abs(np.linalg.norm(d[:, : (bound_dim + 1)], axis=-1) - 1)) < 1e-5
        ), "The hypersphere ssubset part of d must be normalized"
        AbstractSE3Distribution.__init__(self)
        LinBoundedCartProdDiracDistribution.__init__(self, d, w)

    @property
    def input_dim(self):
        return self.dim + 1
