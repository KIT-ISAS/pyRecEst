import numpy as np

from .abstract_ellipsoidal_ball_distribution import AbstractEllipsoidalBallDistribution
from .abstract_uniform_distribution import AbstractUniformDistribution


class EllipsoidalBallUniformDistribution(
    AbstractEllipsoidalBallDistribution, AbstractUniformDistribution
):
    def __init__(self, center, shape_matrix):
        AbstractUniformDistribution.__init__(self)
        AbstractEllipsoidalBallDistribution.__init__(self, center, shape_matrix)

    def mean(self):
        raise NotImplementedError()

    def pdf(self, xs):
        assert xs.shape[-1] == self.dim
        # Calculate the reciprocal of the volume of the ellipsoid
        # reciprocal_volume = 1 / (np.power(np.pi, self.dim / 2) * np.sqrt(np.linalg.det(self.shape_matrix)) / gamma(self.dim / 2 + 1))
        reciprocal_volume = 1 / self.get_manifold_size()
        if xs.ndim == 1:
            return reciprocal_volume
        
        n = xs.shape[0]
        results = np.zeros(n)

        # Check if points are inside the ellipsoid
        for i in range(n):
            point = xs[i, :]
            diff = point - self.center
            result = np.dot(diff.T, np.linalg.solve(self.shape_matrix, diff))

            # If the point is inside the ellipsoid, store the reciprocal of the volume as the pdf value
            if result <= 1:
                results[i] = reciprocal_volume

        return results

    def sample(self, n=1):
        # Generate random points uniformly in a unit d-dimensional ball
        random_points = np.random.randn(n, self.dim)
        random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)

        random_radii = np.random.rand(n, 1)
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
