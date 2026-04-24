# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye

from .abstract_ellipsoidal_ball_distribution import AbstractEllipsoidalBallDistribution


class AbstractDiskDistribution(AbstractEllipsoidalBallDistribution):
    """
    This class represents an abstract base for distributions on the unit disk.
    """

    # We index it using 2-D Euclidean vectors (is zero everywhere else)
    def __init__(self):
        super().__init__(array([0, 0]), eye(2))

    def mean(self):
        raise TypeError("Mean not defined for distributions on the disk.")
