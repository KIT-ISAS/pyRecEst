import numpy as np
from .watson_distribution import WatsonDistribution
from .abstract_hyperhemispherical_distribution import AbstractHyperhemisphericalDistribution

class HyperhemisphericalWatsonDistribution(AbstractHyperhemisphericalDistribution):
    def __init__(self, mu_, kappa_):
        assert mu_[-1] >= 0
        self.distFullSphere = WatsonDistribution(mu_, kappa_)
        self.dim = self.distFullSphere.dim

    def pdf(self, xa):
        return 2 * self.distFullSphere.pdf(xa)

    def set_mode(self, mu):
        w = self
        w.mu = mu
        return w

    def sample(self, n):
        s_full = self.distFullSphere.sample(n)
        s = s_full * (-1) ** (s_full[-1] < 0)  # Mirror to upper hemisphere
        return s

    @property
    def mu(self):
        return self.distFullSphere.mu

    @mu.setter
    def mu(self, mu):
        self.distFullSphere.mu = mu

    @property
    def kappa(self):
        return self.distFullSphere.kappa

    @kappa.setter
    def kappa(self, kappa):
        self.distFullSphere.kappa = kappa

    def mode(self):
        return self.mu

    def shift(self, offsets):
        assert np.allclose(
            self.mu, np.append(np.zeros(self.dim - 1), 1)
        ), "There is no true shifting for the hyperhemisphere. This is a function for compatibility and only works when mu is [0,0,...,1]."
        dist_shifted = self
        dist_shifted.mu = offsets
        return dist_shifted
