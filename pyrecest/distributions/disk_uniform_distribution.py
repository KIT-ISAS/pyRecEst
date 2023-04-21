import numpy as np

from .abstract_disk_distribution import AbstractDiskDistribution
from .ellipsoidal_ball_uniform_distribution import EllipsoidalBallUniformDistribution


class DiskUniformDistribution(
    EllipsoidalBallUniformDistribution, AbstractDiskDistribution
):
    def __init__(self):
        EllipsoidalBallUniformDistribution.__init__(self, np.array([0, 0]), np.eye(2))
