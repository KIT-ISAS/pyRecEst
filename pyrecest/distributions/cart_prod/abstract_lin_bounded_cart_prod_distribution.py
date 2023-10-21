from typing import Union
from pyrecest.backend import int64
from pyrecest.backend import int32
from abc import abstractmethod


from beartype import beartype

from .abstract_cart_prod_distribution import AbstractCartProdDistribution


class AbstractLinBoundedCartProdDistribution(AbstractCartProdDistribution):
    """
    For Cartesian products of linear and bounded (periodic or parts of
    Euclidean spaces) domains. Assumption is that the input dimensions
    are ordered as follows: bounded dimensions first, then linear dimensions.
    """

    def __init__(
        self, bound_dim: Union[int, int32, int64], lin_dim: Union[int, int32, int64]
    ):
        """
        Parameters:
            bound_dim (int)
                number of bounded (e.g., periodic or hyperrectangular) dimensions
            lin_dim (int)
                number of linear dimensions

        """
        if not bound_dim >= 1:
            raise ValueError("bound_dim must be a positive integer")
        if not bound_dim >= 1:
            raise ValueError("lin_dim must be a positive integer")

        AbstractCartProdDistribution.__init__(self, bound_dim + lin_dim)
        self.bound_dim = bound_dim
        self.lin_dim = lin_dim

    def mean(self):
        """
        Convenient access to hybrid_mean() to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype: 
        """
        return self.hybrid_mean()

    def hybrid_mean(self):
        return (
            self.marginalize_linear().mean(),
            self.marginalize_periodic().mean(),
        )

    @abstractmethod
    def marginalize_linear(self):
        pass

    @abstractmethod
    def marginalize_periodic(self):
        pass