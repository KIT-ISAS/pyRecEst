from pyrecest.backend import array, eye

from .abstract_disk_distribution import AbstractDiskDistribution
from .ellipsoidal_ball_uniform_distribution import EllipsoidalBallUniformDistribution


class DiskUniformDistribution(
    EllipsoidalBallUniformDistribution, AbstractDiskDistribution
):
    """
    A class used to represent uniform distribution on a disk.
    Inherits from EllipsoidalBallUniformDistribution and AbstractDiskDistribution.
    """

    def __init__(self):
        """
        Initialize DiskUniformDistribution.

        The center of the disk is at [0, 0] and the shape matrix of the ellipsoid is an identity covariance matrix.
        """
        AbstractDiskDistribution.__init__(self)
        EllipsoidalBallUniformDistribution.__init__(self, array([0, 0]), eye(2))
