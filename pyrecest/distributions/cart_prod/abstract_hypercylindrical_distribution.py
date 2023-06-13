from abc import abstractmethod

import numpy as np

from ..nonperiodic.custom_linear_distribution import CustomLinearDistribution
from .abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)


class AbstractHypercylindricalDistribution(AbstractLinPeriodicCartProdDistribution):
    def __init__(
        self, bound_dim: int | np.int32 | np.int64, lin_dim: int | np.int32 | np.int64
    ):
        AbstractLinPeriodicCartProdDistribution.__init__(self, bound_dim, lin_dim)

    @abstractmethod
    def pdf(self, xs):
        pass

    def condition_on_periodic(self, input_periodic, normalize=True):
        """
        Conditions the distribution on periodic variables

        Arguments:
            input_periodic: ndarray
                Input data, assumed to have shape (self.bound_dim,)
            normalize: bool
                If True, normalizes the distribution

        Returns:
            dist: CustomLinearDistribution
                CustomLinearDistribution instance
        """
        assert (
            input_periodic.shape == (self.bound_dim,)
            or len(input_periodic.shape) == 0
            and self.bound_dim == 1
        )

        def f_cond_unnorm(x):
            input_repeated = np.tile(input_periodic, (1, x.shape[1]))
            return self.pdf(np.vstack((input_repeated, x)))

        dist = CustomLinearDistribution(f_cond_unnorm, self.lin_dim)

        if normalize:  # Conditional may not be normalized
            dist = dist.normalize()

        return dist

    @property
    def input_dim(self):
        return self.dim
