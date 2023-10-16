from pyrecest.backend import sqrt
import numbers

import numpy as np
from beartype import beartype
from scipy.special import gamma

from .abstract_bounded_nonperiodic_distribution import (
    AbstractBoundedNonPeriodicDistribution,
)


class AbstractEllipsoidalBallDistribution(AbstractBoundedNonPeriodicDistribution):
    """
    This class represents distributions on ellipsoidal balls.
    """

    @beartype
    def __init__(self, center: np.ndarray, shape_matrix: np.ndarray):
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

    @beartype
    def get_manifold_size(self) -> np.number | numbers.Real:
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
            c = np.pi
        elif self.dim == 3:
            c = 4 / 3 * np.pi
        elif self.dim == 4:
            c = 0.5 * np.pi**2
        else:
            c = (np.pi ** (self.dim / 2)) / gamma((self.dim / 2) + 1)

        return c * sqrt(np.linalg.det(self.shape_matrix))
