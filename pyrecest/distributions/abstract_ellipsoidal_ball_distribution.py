from math import pi

from pyrecest.backend import linalg, sqrt
from scipy.special import gamma

from .abstract_bounded_nonperiodic_distribution import (
    AbstractBoundedNonPeriodicDistribution,
)


class AbstractEllipsoidalBallDistribution(AbstractBoundedNonPeriodicDistribution):
    """
    This class represents distributions on ellipsoidal balls.
    """

    def __init__(self, center, shape_matrix):
        """
        Initialize the class with a center and shape matrix.

        :param center: The center of the ellipsoidal ball.
        :param shape_matrix: The shape matrix of the ellipsoidal ball.
        """
        AbstractBoundedNonPeriodicDistribution.__init__(self, center.shape[-1])
        self.center = center
        self.shape_matrix = shape_matrix
        assert center.ndim == 1 and shape_matrix.ndim == 2
        assert shape_matrix.shape[0] == self.dim and shape_matrix.shape[1] == self.dim

    def get_manifold_size(self):
        """
        Calculate the size of the manifold.

        :returns: The size of the manifold.
        """
        # Handle cases with dimensions up to 4 directly
        if self.dim == 0:
            return 1

        if self.dim == 1:
            c = 2
        elif self.dim == 2:
            c = pi
        elif self.dim == 3:
            c = 4 / 3 * pi
        elif self.dim == 4:
            c = 0.5 * pi**2
        else:
            c = (pi ** (self.dim / 2)) / gamma((self.dim / 2) + 1)

        return c * sqrt(linalg.det(self.shape_matrix))
