from math import pi
from typing import Union

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    arange,
    array,
    cos,
    floor,
    int32,
    int64,
    linspace,
    mod,
    random,
    sin,
    sum,
)

from ..abstract_grid_distribution import AbstractGridDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalGridDistribution(
    AbstractGridDistribution, AbstractHypertoroidalDistribution
):
    """
    Grid distribution on the hypertorus. Stores values at equidistant grid points
    on [0, 2*pi)^dim. Currently only dim=1 (circle) is fully supported.
    """

    def __init__(self, grid_values, dim: Union[int, int32, int64] = 1):
        """
        Parameters
        ----------
        grid_values : array-like
            Values at the equidistant grid points. Shape (n,) for dim=1.
        dim : int
            Dimension of the hypertorus (default 1).
        """
        assert dim == 1, "Currently only dim=1 (circle) is supported."
        grid_values = np.asarray(grid_values, dtype=float)
        n = grid_values.shape[0]
        grid = np.linspace(0.0, 2.0 * pi, n, endpoint=False)
        AbstractGridDistribution.__init__(
            self, grid_values, grid_type="equidistant", grid=grid, dim=dim
        )
        AbstractHypertoroidalDistribution.__init__(self, dim)

    @property
    def n_grid_points(self):
        return self.grid_values.shape[0]

    def get_closest_point(self, xs):
        """
        Find the index of the closest grid point for each query point.

        Parameters
        ----------
        xs : array-like
            Query points, shape (n,) for dim=1.

        Returns
        -------
        grid_points : ndarray
            The closest grid points.
        indices : ndarray of int
            The indices of the closest grid points.
        """
        n = self.n_grid_points
        xs_mod = np.mod(np.asarray(xs, dtype=float).ravel(), 2.0 * pi)
        indices = (np.floor(xs_mod * n / (2.0 * pi) + 0.5) % n).astype(int)
        return self.grid[indices], indices

    def get_grid_point(self, idx):
        """Return the grid point at index idx as a 1-D array."""
        return array([self.grid[idx]])

    def get_manifold_size(self):
        return (2.0 * pi) ** self.dim

    # Override so we use the grid-based trigonometric moment for efficiency
    def trigonometric_moment(self, n: Union[int, int32, int64]):
        """Compute the n-th trigonometric moment from grid values."""
        grid = self.grid
        w = self.grid_values / np.sum(self.grid_values)
        # Riemann sum approximation: sum_i w_i * exp(i*n*theta_i) * (2*pi/N) * N/(2*pi) = sum_i w_i * exp(i*n*theta_i)
        # But since grid_values already encodes the weights proportional to the pdf,
        # we compute: sum_i grid_values[i] * exp(i*n*theta[i]) / sum_i grid_values[i]
        return np.sum(w * np.exp(1j * n * grid))

    def mean_direction(self):
        """Compute the mean direction from the grid values."""
        m1 = self.trigonometric_moment(1)
        return np.mod(np.angle(m1), 2.0 * pi)

    def mode(self):
        """Return the grid point with the highest weight."""
        idx = int(np.argmax(self.grid_values))
        return array([self.grid[idx]])

    def sample(self, n: Union[int, int32, int64]):
        """Sample n points from the distribution using weighted sampling."""
        w = self.grid_values / np.sum(self.grid_values)
        indices = np.random.choice(len(w), size=n, p=w)
        return array(self.grid[indices])

    def pdf(self, xs):
        """Evaluate the pdf at xs using nearest-neighbor interpolation."""
        _, indices = self.get_closest_point(xs)
        n = self.n_grid_points
        normalization = np.sum(self.grid_values) * (2.0 * pi) / n
        return array(self.grid_values[indices] / normalization)

    def integrate(self, integration_boundaries=None):
        """Integrate the distribution over the full circle."""
        assert integration_boundaries is None, (
            "Custom integration boundaries are not supported."
        )
        n = self.n_grid_points
        return float(np.sum(self.grid_values) * (2.0 * pi) / n)

    @staticmethod
    def from_distribution(dist, n: Union[int, int32, int64]):
        """
        Create a HypertoroidalGridDistribution by sampling a distribution at
        equidistant grid points.

        Parameters
        ----------
        dist : AbstractHypertoroidalDistribution
            The distribution to approximate.
        n : int
            Number of grid points.

        Returns
        -------
        HypertoroidalGridDistribution
        """
        grid = np.linspace(0.0, 2.0 * pi, n, endpoint=False)
        grid_values = np.asarray(dist.pdf(array(grid)), dtype=float)
        return HypertoroidalGridDistribution(grid_values, dim=dist.dim)
