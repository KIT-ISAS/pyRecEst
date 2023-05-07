import mpmath
import numpy as np
from scipy.linalg import qr

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution


class WatsonDistribution(AbstractHypersphericalDistribution):
    def __init__(self, mu_, kappa_):
        AbstractHypersphericalDistribution.__init__(self, dim=mu_.shape[0] - 1)
        epsilon = 1e-6
        assert mu_.ndim == 1, "mu must be a 1-D vector"
        assert np.abs(np.linalg.norm(mu_) - 1) < epsilon, "mu is unnormalized"

        self.mu = mu_
        self.kappa = kappa_

        C_mpf = (
            mpmath.gamma((self.dim + 1) / 2)
            / (2 * mpmath.pi ** ((self.dim + 1) / 2))
            / mpmath.hyper([0.5], [(self.dim + 1) / 2.0], self.kappa)
        )
        self.C = np.float64(C_mpf)

    def pdf(self, xs):
        assert xs.shape[-1] == self.dim + 1
        p = self.C * np.exp(self.kappa * (self.mu.T @ xs.T) ** 2)
        return p

    def to_bingham(self):
        if self.kappa < 0:
            raise NotImplementedError(
                "Conversion to Bingham is not implemented for kappa<0"
            )

        M = np.tile(self.mu, (1, self.dim + 1))
        E = np.eye(self.dim + 1)
        E[0, 0] = 0
        M = M + E
        Q, _ = qr(M)
        M = np.hstack([Q[:, 1:], Q[:, 0].reshape(-1, 1)])
        Z = np.vstack([np.full((self.dim, 1), -self.kappa), 0])
        return BinghamDistribution(Z, M)

    def sample(self, n):
        if self.dim != 2:
            return self.to_bingham().sample(n)

        return super().sample(n)

    def mode(self):
        if self.kappa >= 0:
            return self.mu

        return self.mode_numerical()

    def set_mode(self, new_mode):
        assert new_mode.shape == self.mu.shape
        dist = self
        dist.mu = new_mode
        return dist

    def shift(self, offsets):
        assert np.array_equal(
            self.mu, np.vstack([np.zeros((self.dim, 1)), 1])
        ), "There is no true shifting for the hypersphere. This is a function for compatibility and only works when mu is [0,0,...,1]."
        return self.set_mode(offsets)
