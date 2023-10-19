from pyrecest.backend import random
from typing import Union
from pyrecest.backend import squeeze
from pyrecest.backend import sqrt
from pyrecest.backend import reshape
from pyrecest.backend import ndim
from pyrecest.backend import meshgrid
from pyrecest.backend import linspace
from pyrecest.backend import array
from pyrecest.backend import int64
from pyrecest.backend import int32
import numbers
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from pyrecest.utils.plotting import plot_ellipsoid
from scipy.integrate import dblquad, nquad, quad
from scipy.optimize import minimize
from scipy.stats import chi2

from ..abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)
from pyrecest.backend import empty, ones


class AbstractLinearDistribution(AbstractManifoldSpecificDistribution):
    @property
    def input_dim(self):
        return self.dim

    def mean(self):
        return self.mean_numerical()

    def covariance(self):
        return self.covariance_numerical()

    def get_manifold_size(self):
        return np.inf

    def mode(self, starting_point=None):
        return self.mode_numerical(starting_point)

    def mode_numerical(self, starting_point=None):
        if starting_point is None:
            # Ensure 1-D for minimize
            starting_point = squeeze(self.sample(1))

        def neg_pdf(x):
            return -self.pdf(x)

        assert ndim(starting_point) <= 1, "Starting point must be a 1D array"
        starting_point = np.atleast_1d(
            starting_point
        )  # Avoid numpy warning "DeprecationWarning: Use of `minimize` with `x0.ndim != 1` is deprecated"

        result = minimize(neg_pdf, starting_point, method="L-BFGS-B")
        return result.x

    def sample_metropolis_hastings(
        self,
        n: Union[int, int32, int64],
        burn_in: Union[int, int32, int64] = 10,
        skipping: Union[int, int32, int64] = 5,
        proposal: Callable | None = None,
        start_point: np.number | numbers.Real | np.ndarray | None = None,
    ) -> np.ndarray:
        if proposal is None:

            def proposal(x):
                return x + random.randn(self.dim)

        if start_point is None:
            start_point = (
                self.mean()
            )  # We assume it is cheaply available. Done so for a lack of a better choice.

        # pylint: disable=duplicate-code
        return AbstractManifoldSpecificDistribution.sample_metropolis_hastings(
            self,
            n,
            burn_in=burn_in,
            skipping=skipping,
            proposal=proposal,
            start_point=start_point,
        )

    def mean_numerical(self):
        if self.dim == 1:
            mu = quad(lambda x: x * self.pdf(x), -np.inf, np.inf)[0]
        elif self.dim == 2:
            mu = array([np.NaN, np.NaN])
            mu[0] = dblquad(
                lambda x, y: x * self.pdf(array([x, y])),
                -np.inf,
                np.inf,
                lambda _: -np.inf,
                lambda _: np.inf,
            )[0]
            mu[1] = dblquad(
                lambda x, y: y * self.pdf(array([x, y])),
                -np.inf,
                np.inf,
                lambda _: -np.inf,
                lambda _: np.inf,
            )[0]
        elif self.dim == 3:
            mu = array([np.NaN, np.NaN, np.NaN])

            def integrand1(x, y, z):
                return x * self.pdf(array([x, y, z]))

            def integrand2(x, y, z):
                return y * self.pdf(array([x, y, z]))

            def integrand3(x, y, z):
                return z * self.pdf(array([x, y, z]))

            mu[0] = nquad(
                integrand1, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
            )[0]
            mu[1] = nquad(
                integrand2, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
            )[0]
            mu[2] = nquad(
                integrand3, [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]
            )[0]
        else:
            raise ValueError(
                "Dimension currently not supported for all types of densities."
            )
        return mu

    def covariance_numerical(self):
        mu = self.mean()
        if self.dim == 1:
            C = quad(lambda x: (x - mu) ** 2 * self.pdf(x), -np.inf, np.inf)[0]
        elif self.dim == 2:
            C = array([[np.NaN, np.NaN], [np.NaN, np.NaN]])

            def integrand1(x, y):
                return (x - mu[0]) ** 2 * self.pdf(array([x, y]))

            def integrand2(x, y):
                return (x - mu[0]) * (y - mu[1]) * self.pdf(array([x, y]))

            def integrand3(x, y):
                return (y - mu[1]) ** 2 * self.pdf(array([x, y]))

            C[0, 0] = nquad(integrand1, [[-np.inf, np.inf], [-np.inf, np.inf]])[0]
            C[0, 1] = nquad(integrand2, [[-np.inf, np.inf], [-np.inf, np.inf]])[0]
            C[1, 0] = C[0, 1]
            C[1, 1] = nquad(integrand3, [[-np.inf, np.inf], [-np.inf, np.inf]])[0]
        else:
            raise NotImplementedError(
                "Covariance numerical not supported for this dimension."
            )
        return C

    def integrate(self, left=None, right=None):
        if left is None:
            left = -np.inf * ones(self.dim)
        if right is None:
            right = np.inf * ones(self.dim)

        result = self.integrate_numerically(left, right)
        return result

    def integrate_numerically(self, left=None, right=None):
        if left is None:
            left = empty(self.dim)
            left[:] = -float('inf')
        if right is None:
            right = empty(self.dim)
            right[:] = float('inf')
        return AbstractLinearDistribution.integrate_fun_over_domain(
            self.pdf, self.dim, left, right
        )

    @staticmethod
    def integrate_fun_over_domain(f, dim, left, right):
        def f_for_nquad(*args):
            # Avoid DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future.
            return squeeze(f(array(args).reshape(-1, dim)))

        if dim == 1:
            result, _ = quad(f, left, right)
        elif dim == 2:
            result, _ = nquad(f_for_nquad, [(left[0], right[0]), (left[1], right[1])])
        elif dim == 3:
            result, _ = nquad(
                f_for_nquad,
                [(left[0], right[0]), (left[1], right[1]), (left[2], right[2])],
            )
        else:
            raise ValueError("Dimension not supported.")
        return result

    def get_suggested_integration_limits(self, scaling_factor=10):
        """
        Returns suggested limits for integration over the whole density.

        The linear part should be integrated from -Inf to Inf but
        Python's numerical integration does not handle that well.
        When we can obtain the covariance of the linear part easily,
        we integrate from mu-10*sigma to mu+scaling_factor*sigma,
        which contains almost the entire probability mass. The
        circular part is integrated form 0 to 2pi.

        Returns:
            l (numpy.ndarray): lower integration bound (shape: (linD+boundD,))
            r (numpy.ndarray): upper integration bound (shape: (linD+boundD,))
        """
        C = self.covariance()
        m = self.mode()
        left = np.full((self.dim,), np.nan)
        right = np.full((self.dim,), np.nan)

        for i in range(self.dim):  # Change for linear dimensions
            left[i] = m[i] - scaling_factor * sqrt(C[i, i])
            right[i] = m[i] + scaling_factor * sqrt(C[i, i])

        return left, right

    def plot(self, *args, plot_range=None, **kwargs):
        mu = self.mean()
        C = self.covariance()

        if plot_range is None:
            scaling = sqrt(chi2.ppf(0.99, self.dim))
            plot_range = empty(2 * self.dim)
            for i in range(0, 2 * self.dim, 2):
                plot_range[i] = mu[int(i / 2)] - scaling * sqrt(
                    C[int(i / 2), int(i / 2)]
                )
                plot_range[i + 1] = mu[int(i / 2)] + scaling * sqrt(
                    C[int(i / 2), int(i / 2)]
                )

        if self.dim == 1:
            x = linspace(plot_range[0], plot_range[1], 1000)
            y = self.pdf(x)
            plt.plot(x, y, *args, **kwargs)
            plt.show()
        elif self.dim == 2:
            x = linspace(plot_range[0], plot_range[1], 100)
            y = linspace(plot_range[2], plot_range[3], 100)
            x_grid, y_grid = meshgrid(x, y)
            z_grid = self.pdf(np.column_stack((x_grid.ravel(), y_grid.ravel())))

            ax = plt.axes(projection="3d")
            ax.plot_surface(
                x_grid, y_grid, reshape(z_grid, x_grid.shape), *args, **kwargs
            )
            plt.show()
        else:
            raise ValueError("Dimension not supported")

    def plot_state(self, scaling_factor=1, color=(0, 0.4470, 0.7410)):
        if self.dim in (
            2,
            3,
        ):
            covariance = self.covariance()
            mean = self.mean()
            plot_ellipsoid(mean, covariance, scaling_factor, color)

        raise ValueError("Dimension currently not supported for plotting the state.")