import numpy as np
from scipy.linalg import qr
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
import mpmath

class WatsonDistribution(AbstractHypersphericalDistribution):
    def __init__(self, mu_, kappa_):
        epsilon = 1e-6
        assert mu_.ndim == 1, 'mu must be a 1-D vector'
        assert np.abs(np.linalg.norm(mu_) - 1) < epsilon, 'mu is unnormalized'

        self.mu = mu_
        self.kappa = kappa_

        self.dim = mu_.shape[0]
        C_mpf = mpmath.gamma(self.dim / 2) / (2 * mpmath.pi ** (self.dim / 2)) / mpmath.hyper([0.5],[self.dim/2.0], self.kappa)
        self.C = np.float64(C_mpf)

    def pdf(self, xa):
        assert xa.shape[0] == self.dim
        p = self.C * np.exp(self.kappa * (self.mu.T @ xa) ** 2)
        return p

    def to_bingham(self):
        if self.kappa < 0:
            raise NotImplementedError('Conversion to Bingham is not implemented for kappa<0')

        M = np.tile(self.mu, (1, self.dim))
        E = np.eye(self.dim)
        E[0, 0] = 0
        M = M + E
        Q, _ = qr(M)
        M = np.hstack([Q[:, 1:], Q[:, 0].reshape(-1, 1)])
        Z = np.vstack([np.full((self.dim - 1, 1), -self.kappa), 0])
        return BinghamDistribution(Z, M)

    def sample(self, n):
        if self.dim != 3:
            return self.to_bingham().sample(n)
        else:
            pass

    def mode(self):
        if self.kappa >= 0:
            return self.mu
        else:
            return self.mode_numerical()

    def set_mode(self, new_mode):
        assert new_mode.shape == self.mu.shape
        dist = self
        dist.mu = new_mode
        return dist

    def shift(self, offsets):
        assert np.array_equal(self.mu, np.vstack([np.zeros((self.dim - 1, 1)), 1])), 'There is no true shifting for the hypersphere. This is a function for compatibility and only works when mu is [0,0,...,1].'
        return self.set_mode(offsets)
