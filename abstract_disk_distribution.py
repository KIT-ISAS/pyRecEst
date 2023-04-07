import numpy as np
from abstract_ellipsoidal_ball_distribution import AbstractEllipsoidalBallDistribution

class AbstractDiskDistribution(AbstractEllipsoidalBallDistribution):
    # We index it using 2-D Euclidean vectors (is zero everywhere else)
    def __init__(self):
        self.dim = 2
        super().__init__(np.array([0, 0]), np.eye(2))

    def mean(self):
        raise TypeError("Mean not defined for distributions on the disk.")
