from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_cart_prod_distribution import AbstractCartProdDistribution


class CartProdDiracDistribution(
    AbstractDiracDistribution, AbstractCartProdDistribution
):
    pass
