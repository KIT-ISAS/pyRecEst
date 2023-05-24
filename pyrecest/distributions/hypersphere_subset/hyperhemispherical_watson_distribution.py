import numpy as np

from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .watson_distribution import WatsonDistribution


class HyperhemisphericalWatsonDistribution(AbstractHyperhemisphericalDistribution):
    def __init__(self, mu: np.array, kappa: float):
        assert mu[-1] >= 0
        self.dist_full_sphere = WatsonDistribution(mu, kappa)
        AbstractHyperhemisphericalDistribution.__init__(
            self, dim=self.dist_full_sphere.dim
        )

    def pdf(self, xs: np.array):
        return 2 * self.dist_full_sphere.pdf(xs)

    def set_mode(self, mu: np.array):
        w = self
        w.mu = mu
        return w

    def sample(self, n: int):
        s_full = self.dist_full_sphere.sample(n)
        s = s_full * (-1) ** (s_full[-1] < 0)  # Mirror to upper hemisphere
        return s

    @property
    def mu(self):
        return self.dist_full_sphere.mu

    @mu.setter
    def mu(self, mu):
        self.dist_full_sphere.mu = mu

    @property
    def kappa(self):
        return self.dist_full_sphere.kappa

    @kappa.setter
    def kappa(self, kappa):
        self.dist_full_sphere.kappa = kappa

    def mode(self):
        return self.mu

    def shift(self, offsets):
        assert np.allclose(
            self.mu, np.append(np.zeros(self.dim - 1), 1)
        ), "There is no true shifting for the hyperhemisphere. This is a function for compatibility and only works when mu is [0,0,...,1]."
        dist_shifted = self
        dist_shifted.mu = offsets
        return dist_shifted
