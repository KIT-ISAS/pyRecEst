from abc import abstractmethod

import numpy as np
import scipy.integrate
import scipy.optimize

from ..hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)
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

    def condition_on_linear(self, input_lin, normalize=True):
        """
        Condition on linear.

        Parameters:
        lin_input : ndarray
            Input array.
        normalize : bool, optional
            If True (default), normalizes the distribution.

        Returns:
        dist : CustomHypertoroidalDistribution
            The distribution after conditioning.
        """
        assert (
            np.size(input_lin) == self.lin_dim and np.ndim(input_lin) <= 1
        ), "Input should be of size (lin_dim,)."

        def f_cond_unnorm(x, input_lin=input_lin):
            n_inputs = np.size(x) // x.shape[-1] if np.ndim(x) > 1 else np.size(x)
            input_repeated = np.tile(input_lin, (n_inputs, 1))
            return self.pdf(np.column_stack((x, input_repeated)))

        dist = CustomHypertoroidalDistribution(f_cond_unnorm, self.bound_dim)

        if normalize:  # Conditional need not be normalized
            dist = dist.normalize()

        return dist

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
            np.size(input_periodic) == self.bound_dim and np.ndim(input_periodic) <= 1
        ), "Input should be of size (lin_dim,)."

        input_periodic = np.mod(input_periodic, 2 * np.pi)

        def f_cond_unnorm(x, input_periodic=input_periodic):
            n_inputs = np.size(x) // x.shape[-1] if np.ndim(x) > 1 else np.size(x)
            input_repeated = np.tile(input_periodic, (n_inputs, 1))
            return self.pdf(np.column_stack((input_repeated, x)))

        dist = CustomLinearDistribution(f_cond_unnorm, self.lin_dim)

        if normalize:  # Conditional may not be normalized
            dist = dist.normalize()

        return dist

    def linear_mean_numerical(self):
        # Define the integrands for the mean calculation
        if self.lin_dim == 1 and self.bound_dim == 1:
            mu = scipy.integrate.nquad(
                lambda x, y: y * self.pdf([x, y]), [[0, 2 * np.pi], [-np.inf, np.inf]]
            )[0]
        elif self.bound_dim == 2 and self.lin_dim == 1:
            mu = scipy.integrate.nquad(
                lambda x, y, z: z * self.pdf([x, y, z]),
                [[0, 2 * np.pi], [0, 2 * np.pi], [-np.inf, np.inf]],
            )[0]
        elif self.bound_dim == 1 and self.lin_dim == 2:
            mu = np.empty(2)
            mu[0] = scipy.integrate.nquad(
                lambda x, y, z: y * self.pdf([x, y, z]),
                [[0, 2 * np.pi], [-np.inf, np.inf], [-np.inf, np.inf]],
            )[0]
            mu[1] = scipy.integrate.nquad(
                lambda x, y, z: z * self.pdf([x, y, z]),
                [[0, 2 * np.pi], [-np.inf, np.inf], [-np.inf, np.inf]],
            )[0]
        else:
            raise ValueError("Cannot determine linear mean for this dimension.")

        return mu

    def mode_numerical(self, starting_point=None):
        """
        Find the mode of the distribution numerically.

        Parameters:
        starting_point : ndarray, optional
            The starting point for the optimization. If None, uses [pi*ones(this.boundD,1);zeros(this.linD,1)]

        Returns:
        m : ndarray
            The mode of the distribution.
        """
        if starting_point is None:
            starting_point = np.concatenate(
                [np.pi * np.ones((self.bound_dim)), np.zeros((self.lin_dim))]
            )

        bounds = [
            (0, 2 * np.pi) if i < self.bound_dim else (-np.inf, np.inf)
            for i in range(len(starting_point))
        ]

        m = scipy.optimize.minimize(
            lambda x: -self.pdf(x), starting_point, bounds=bounds
        ).x

        return m

    @property
    def input_dim(self):
        return self.dim
