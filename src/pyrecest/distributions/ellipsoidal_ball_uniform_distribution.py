from typing import Union

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import int32, int64, linalg, random, where, zeros

from .abstract_ellipsoidal_ball_distribution import AbstractEllipsoidalBallDistribution
from .abstract_uniform_distribution import AbstractUniformDistribution


class EllipsoidalBallUniformDistribution(
    AbstractEllipsoidalBallDistribution, AbstractUniformDistribution
):
    """A class representing a uniform distribution on an ellipsoidal ball."""

    def __init__(self, center, shape_matrix):
        """
        Initialize EllipsoidalBallUniformDistribution.

        :param center: Center of the ellipsoidal ball.
        :param shape_matrix: Shape matrix defining the ellipsoidal ball.
        """
        AbstractUniformDistribution.__init__(self)
        AbstractEllipsoidalBallDistribution.__init__(self, center, shape_matrix)

    @property
    def input_dim(self) -> int:
        """Returns the size of the input vector for evaluation of the pdf."""
        return self.dim

    def mean(self):
        raise NotImplementedError()

    def pdf(self, xs):
        """
        Compute the probability density function at given points.

        :param xs: Points at which to compute the PDF.
        :returns: PDF values at given points.
        """
        assert xs.shape[-1] == self.dim

        reciprocal_volume = 1 / self.get_manifold_size()

        # Make xs always 2D for uniform handling
        single = xs.ndim == 1
        if single:
            xs = xs[None, :]

        # (n, dim)
        diff = xs - self.center[None, :]

        # Solve S * y = diff^T  -> y^T = diff^T * S^{-1}
        # S: (dim, dim), diff.T: (dim, n)  => solved.T: (n, dim)
        solved = linalg.solve(self.shape_matrix, diff.T).T

        # Quadratic form per row: sum_i diff_i * solved_i
        quad = (diff * solved).sum(axis=1)

        # Optional tiny tolerance near the boundary:
        inside = quad <= 1.0

        pdf_values = where(inside, reciprocal_volume, zeros(quad.shape[0]))

        return pdf_values[0] if single else pdf_values

    def sample(self, n: Union[int, int32, int64]):
        """
        Generate samples from the distribution.

        :param n: Number of samples to generate.
        :returns: Generated samples.
        """
        random_points = random.normal(size=(n, self.dim))
        random_points /= linalg.norm(random_points, axis=1).reshape(-1, 1)

        random_radii = random.uniform(size=(n, 1))  # So that broadcasting works below
        random_radii = random_radii ** (
            1 / self.dim
        )  # Consider that the ellipsoid surfaces with higher radii are larger

        # Scale random points by the radii
        random_points *= random_radii

        # Rotate the points according to the shape matrix
        L = linalg.cholesky(self.shape_matrix)
        # For points (d, n), this would be L @ random_points
        transformed_points = random_points @ L.T + self.center.reshape(1, -1)

        return transformed_points
