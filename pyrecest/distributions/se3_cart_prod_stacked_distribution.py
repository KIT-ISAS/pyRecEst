import numpy as np

from .abstract_se3_distribution import AbstractSE3Distribution
from .cart_prod.cart_prod_stacked_distribution import CartProdStackedDistribution


class SE3CartProdStackedDistribution(
    CartProdStackedDistribution, AbstractSE3Distribution
):
    def __init__(self, dists):
        AbstractSE3Distribution.__init__(self)
        CartProdStackedDistribution.__init__(self, dists)

    def marginalize_linear(self):
        return self.dists[0]

    def marginalize_periodic(self):
        return self.dists[1]

    def get_manifold_size(self):
        return np.inf

    def pdf(self, xs):
        return CartProdStackedDistribution.pdf(self, xs)