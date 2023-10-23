from abc import abstractmethod
from math import pi
from typing import Union

import scipy.integrate
import scipy.optimize
# pylint: disable=redefined-builtin
from pyrecest.backend import (
    allclose,
    any,
    array,
    column_stack,
    concatenate,
    empty,
    full,
    int32,
    int64,
    isnan,
    mod,
    ndim,
    ones,
    sqrt,
    tile,
    vstack,
    zeros,
)
from scipy.integrate import nquad

from ..hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)
from ..nonperiodic.custom_linear_distribution import CustomLinearDistribution
from .abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)


class AbstractHypercylindricalDistribution(AbstractLinPeriodicCartProdDistribution):
    def __init__(
        self, bound_dim: Union[int, int32, int64], lin_dim: Union[int, int32, int64]
    ):
        AbstractLinPeriodicCartProdDistribution.__init__(self, bound_dim, lin_dim)

    @abstractmethod
    def pdf(self, xs):
        pass

    def integrate(self, integration_boundaries=None):
        return self.integrate_numerically(integration_boundaries)

    def integrate_numerically(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.get_reasonable_integration_boundaries()

        def f(*args):
            return self.pdf(array(args))

        integration_result = nquad(f, integration_boundaries)[0]

        return integration_result

    def get_reasonable_integration_boundaries(self, scalingFactor=10):
        """
        Returns reasonable integration boundaries for the specific distribution
        based on the mode and covariance.
        """
        left = empty((self.bound_dim + self.lin_dim, 1))
        right = empty((self.bound_dim + self.lin_dim, 1))
        P = self.linear_covariance()
        m = self.mode()

        for i in range(self.bound_dim, self.bound_dim + self.lin_dim):
            left[i] = m[i] - scalingFactor * sqrt(
                P[i - self.bound_dim, i - self.bound_dim]
            )
            right[i] = m[i] + scalingFactor * sqrt(
                P[i - self.bound_dim, i - self.bound_dim]
            )

        return vstack((left, right))

    def mode(self):
        """Find the mode of the distribution by calling mode_numerical."""
        return self.mode_numerical()

    def linear_covariance(self, approximate_mean=None):
        """
        Calculates the linear covariance, or calls linear_covariance_numerical
        if a non-numerical solution doesn't exist.

        Parameters:
        - approximate_mean : ndarray, optional
          The approximate mean to be used. If None, uses NaNs to flag for calculation.

        Returns:
        - C : ndarray
          The linear covariance.
        """
        if approximate_mean is None:
            approximate_mean = full((self.lin_dim,), float("NaN"))

        assert approximate_mean.shape[0] == self.lin_dim

        return self.linear_covariance_numerical(approximate_mean)

    def linear_covariance_numerical(self, approximate_mean=None):
        """
        Numerically calculates the linear covariance.

        Parameters:
        - approximate_mean : ndarray, optional
          The approximate mean to be used. If None, calculates the mean.

        Returns:
        - C : ndarray
          The linear covariance.
        """
        if approximate_mean is None or any(isnan(approximate_mean)):
            approximate_mean = self.linear_mean_numerical()

        if self.bound_dim == 1 and self.lin_dim == 1:
            C, _ = nquad(
                lambda x, y: (y - approximate_mean) ** 2 * self.pdf([x, y]),
                [[0.0, 2.0 * pi], [-float("inf"), float("inf")]],
            )
        elif self.bound_dim == 2 and self.lin_dim == 1:
            C, _ = nquad(
                lambda x, y, z: (z - approximate_mean) ** 2 * self.pdf([x, y, z]),
                [[0.0, 2.0 * pi], [0.0, 2.0 * pi], [-float("inf"), float("inf")]],
            )
        elif self.bound_dim == 1 and self.lin_dim == 2:
            C = empty((2, 2))
            C[0, 0], _ = nquad(
                lambda x, y, z: (y - approximate_mean[0]) ** 2 * self.pdf([x, y, z]),
                [
                    [0.0, 2.0 * pi],
                    [-float("inf"), float("inf")],
                    [-float("inf"), float("inf")],
                ],
            )
            C[0, 1], _ = nquad(
                lambda x, y, z: (y - approximate_mean[0])
                * (z - approximate_mean[1])
                * self.pdf([x, y, z]),
                [
                    [0, 2 * pi],
                    [-float("inf"), float("inf")],
                    [-float("inf"), float("inf")],
                ],
            )
            C[1, 0] = C[0, 1]
            C[1, 1], _ = nquad(
                lambda x, y, z: (z - approximate_mean[1]) ** 2 * self.pdf([x, y, z]),
                [
                    [0.0, 2.0 * pi],
                    [-float("inf"), float("inf")],
                    [-float("inf"), float("inf")],
                ],
            )
        else:
            raise ValueError("Cannot determine linear covariance for this dimension.")

        return C

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
            input_lin.ndim == 0
            and self.lin_dim == 1
            or ndim(input_lin) == 1
            and input_lin.shape[0] == self.lin_dim
        ), "Input should be of size (lin_dim,)."

        def f_cond_unnorm(x, input_lin=input_lin):
            n_inputs = x.shape[0] if x.ndim > 1 else 1
            input_repeated = tile(input_lin, (n_inputs, 1))
            return self.pdf(column_stack((x, input_repeated)))

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
            input_periodic.ndim == 0
            or input_periodic.shape[0] == self.bound_dim
            and ndim(input_periodic) == 2
        ), "Input should be of size (lin_dim,)."

        input_periodic = mod(input_periodic, 2.0 * pi)

        def f_cond_unnorm(x, input_periodic=input_periodic):
            n_inputs = x.shape[0] if x.ndim > 1 else 1
            input_repeated = tile(input_periodic, (n_inputs, 1))
            return self.pdf(column_stack((input_repeated, x)))

        dist = CustomLinearDistribution(f_cond_unnorm, self.lin_dim)

        if normalize:  # Conditional may not be normalized
            dist = dist.normalize()

        return dist

    def linear_mean_numerical(self):
        # Define the integrands for the mean calculation
        if self.lin_dim == 1 and self.bound_dim == 1:
            mu = scipy.integrate.nquad(
                lambda x, y: (y * self.pdf(array([x, y])))[0],
                [[0.0, 2 * pi], [-float("inf"), float("inf")]],
            )[0]
        elif self.bound_dim == 2 and self.lin_dim == 1:
            mu = scipy.integrate.nquad(
                lambda x, y, z: (z * self.pdf([x, y, z]))[0],
                [[0.0, 2 * pi], [0.0, 2 * pi], [-float("inf"), float("inf")]],
            )[0]
        elif self.bound_dim == 1 and self.lin_dim == 2:
            mu = empty(2)
            mu[0] = scipy.integrate.nquad(
                lambda x, y, z: (y * self.pdf([x, y, z]))[0],
                [
                    [0.0, 2 * pi],
                    [-float("inf"), float("inf")],
                    [-float("inf"), float("inf")],
                ],
            )[0]
            mu[1] = scipy.integrate.nquad(
                lambda x, y, z: (z * self.pdf([x, y, z]))[0],
                [
                    [0, 2 * pi],
                    [-float("inf"), float("inf")],
                    [-float("inf"), float("inf")],
                ],
            )[0]
        else:
            raise ValueError("Cannot determine linear mean for this dimension.")

        return mu

    def mode_numerical(self, starting_point=None):
        """
        Find the mode of the distribution numerically.

        Parameters:
        starting_point : ndarray, optional
          The starting point for the optimization.
          If None, uses [pi * ones(self.bound_dim); zeros(self.lin_dim)]

        Returns:
        m : ndarray
          The mode of the distribution.
        """
        if starting_point is None:
            starting_point = concatenate(
                [pi * ones(self.bound_dim), zeros(self.lin_dim)]
            )

        # Define bounds for the optimization
        bounds = [
            (0.0, 2.0 * pi) if i < self.bound_dim else (-float("inf"), float("inf"))
            for i in range(self.bound_dim + self.lin_dim)
        ]

        # Perform the optimization
        res = scipy.optimize.minimize(
            lambda x: -self.pdf(array(x)), starting_point, bounds=bounds
        )

        # Check if the optimization might have stopped early
        if allclose(res.x, starting_point):
            print(
                "Warning: Mode was at the starting point. This may indicate the optimizer stopped early."
            )

        return res.x

    @property
    def input_dim(self):
        return self.dim
