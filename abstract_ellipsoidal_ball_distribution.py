from abstract_distribution import AbstractDistribution
import numpy as np
from scipy.special import gamma


class AbstractEllipsoidalBallDistribution(AbstractDistribution):
    def __init__(self, center, shape_matrix):
        self.center = center
        self.shape_matrix = shape_matrix
        self.dim = center.shape[-1]
        assert shape_matrix.shape[0] == self.dim and shape_matrix.shape[1] == self.dim

    def get_manifold_size(self):
        if self.dim == 0:
            return 1
        elif self.dim == 1:
            c = 2
        elif self.dim == 2:
            c = np.pi
        elif self.dim == 3:
            c = 4/3 * np.pi
        elif self.dim == 4:
            c = 0.5 * np.pi**2
        else:
            c = (np.pi**(self.dim/2)) / gamma((self.dim/2) + 1)

        return c * np.sqrt(np.linalg.det(self.shape_matrix))
