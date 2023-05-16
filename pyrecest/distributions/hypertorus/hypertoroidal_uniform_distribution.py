import numpy as np
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from ..abstract_uniform_distribution import AbstractUniformDistribution
import warnings

class HypertoroidalUniformDistribution(AbstractUniformDistribution, AbstractHypertoroidalDistribution):

    def pdf(self, xs):
        return 1/self.get_manifold_size() * np.ones(xs.size//self.dim)

    def trigonometric_moment(self, n: int):
        if n == 0:
            return np.ones(self.dim)
        else:
            return np.zeros(self.dim)

    def entropy(self):
        return self.dim * np.log(2 * np.pi)

    def circular_mean(self):
        warnings.warn('Circular uniform distribution does not have a unique mean', RuntimeWarning)
        return np.nan

    def sample(self, n):
        return 2 * np.pi * np.random.rand(self.dim, n)

    def shift(self, shift_angles):
        assert shift_angles.shape == (self.dim,)
        return self

    def integral(self, l=None, r=None):
        if l is None:
            l = np.zeros((self.dim, 1))
        if r is None:
            r = 2 * np.pi * np.ones((self.dim, 1))
        assert l.shape == (self.dim, 1)
        assert r.shape == (self.dim, 1)

        volume = np.prod(r - l)
        return 1 / (2 * np.pi) ** self.dim * volume
