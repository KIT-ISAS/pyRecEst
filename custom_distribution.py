import numpy as np
from abstract_distribution import AbstractDistribution

class CustomDistribution(AbstractDistribution):

    def __init__(self, f, dim: int, shift_by: np.ndarray = None):
        self.dim = dim
        self.f = f
        self.scale_by = 1

        if shift_by is None:
            shift_by = np.zeros(self.dim)
        assert shift_by.ndim == 1, "Shift_by must be a 1-D numpy vector."
        assert shift_by.shape[0] == self.dim, "Shift_by vector length must match the dimension."
        self.shift_by = shift_by
        
    def pdf(self, xs):
        assert xs.shape[-1] == self.dim
        # Reshape to properly handle the (d,) shape as well as the (n, d) case
        p = self.scale_by * self.f(np.reshape(xs, (-1, self.dim)) - self.shift_by[np.newaxis, :])
        assert p.ndim <= 1,  "Output format of pdf is not as expected"
        return p

    def shift(self, shift_vector):
        assert self.dim == shift_vector.shape[0]
        cd = self
        cd.shift_by = self.shift_by + shift_vector
        return cd

    def normalize(self):
        cd = self
        cd.scale_by = cd.scale_by / self.integral()
        return cd
