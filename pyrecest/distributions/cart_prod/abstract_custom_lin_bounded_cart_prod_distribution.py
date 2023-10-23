from typing import Callable


from ..abstract_custom_distribution import AbstractCustomDistribution
from .abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)


class AbstractCustomLinBoundedCartProdDistribution(
    AbstractCustomDistribution, AbstractLinPeriodicCartProdDistribution
):
    """Is abstract because .input_dim (among others) cannot be properly defined without specifying the specific periodic dimension"""

    def __init__(self, f_: Callable, bound_dim: int, lin_dim: int):
        """
        Parameters:
            f_ (callable)
                pdf of the distribution
            bound_dim (int)
                dimension of the bounded part of the manifold
            lin_dim (int)
                dimension of the linear part of the manifold
        """
        if not bound_dim > 0:
            raise ValueError("bound_dim must be positive")
        if not lin_dim > 0:
            raise ValueError("lin_dim must be positive")

        AbstractCustomDistribution.__init__(self, f_)
        AbstractLinPeriodicCartProdDistribution.__init__(self, bound_dim, lin_dim)

    @staticmethod
    def from_distribution(distribution: AbstractLinPeriodicCartProdDistribution):
        chhd = AbstractCustomLinBoundedCartProdDistribution(
            distribution.pdf, distribution.bound_dim, distribution.lin_dim
        )
        return chhd
