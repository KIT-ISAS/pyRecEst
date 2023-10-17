from pyrecest.backend import random
from typing import Union
from pyrecest.backend import sqrt
from pyrecest.backend import power
from pyrecest.backend import dot
from pyrecest.backend import int64
from pyrecest.backend import int32
from pyrecest.backend import zeros
import numpy as np
from beartype import beartype

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

    def pdf(self, xs: np.ndarray):
        """
        Compute the probability density function at given points.

        :param xs: Points at which to compute the PDF.
        :returns: PDF values at given points.
        """
        assert xs.shape[-1] == self.dim
        # Calculate the reciprocal of the volume of the ellipsoid
        # reciprocal_volume = 1 / (power(np.pi, self.dim / 2) * sqrt(np.linalg.det(self.shape_matrix)) / gamma(self.dim / 2 + 1))
        reciprocal_volume = 1 / self.get_manifold_size()
        if xs.ndim == 1:
            return reciprocal_volume

        n = xs.shape[0]
        results = zeros(n)

        # Check if points are inside the ellipsoid
        for i in range(n):
            point = xs[i, :]
            diff = point - self.center
            result = dot(diff.T, np.linalg.solve(self.shape_matrix, diff))

            # If the point is inside the ellipsoid, store the reciprocal of the volume as the pdf value
            if result <= 1:
                results[i] = reciprocal_volume

        return results

    def sample(self, n: Union[int, int32, int64]) -> np.ndarray:
        """
        Generate samples from the distribution.

        :param n: Number of samples to generate.
        :returns: Generated samples.
        """
        random_points = random.randn(n, self.dim)
        random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)

        random_radii = random.rand(n, 1)
        random_radii = random_radii ** (
            1 / self.dim
        )  # Consider that the ellipsoid surfaces with higher radii are larger

        # Scale random points by the radii
        random_points *= random_radii

        # Rotate the points according to the shape matrix
        L = np.linalg.cholesky(self.shape_matrix)
        # For points (d, n), this would be L @ random_points
        transformed_points = random_points @ L.T + self.center.reshape(1, -1)

        return transformed_points