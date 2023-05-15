import numpy as np
from scipy.linalg import qr
from scipy.special import iv

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class VonMisesFisherDistribution(AbstractHypersphericalDistribution):
    def __init__(self, mu_, kappa_):
        AbstractHypersphericalDistribution.__init__(self, dim=mu_.shape[0] - 1)
        epsilon = 1e-6
        assert (
            mu_.shape[0] >= 2
        ), "mu must be at least two-dimensional for the circular case"
        assert abs(np.linalg.norm(mu_) - 1) < epsilon, "mu must be a normalized"

        self.mu = mu_
        self.kappa = kappa_

        if self.dim == 2:
            self.C = kappa_ / (4 * np.pi * np.sinh(kappa_))
        else:
            self.C = kappa_ ** ((self.dim + 1) / 2 - 1) / (
                (2 * np.pi) ** ((self.dim + 1) / 2) * iv((self.dim + 1) / 2 - 1, kappa_)
            )

    def pdf(self, xs):
        assert xs.shape[-1] == self.input_dim

        return self.C * np.exp(self.kappa * self.mu.T @ xs.T)

    def mean_direction(self):
        return self.mu

    # def sample(self, n):
    # """
    # Generate n von Mises-Fisher distributed random vectors.

    # Parameters:
    # n (int): Number of samples to generate.

    # Returns:
    # array: n von Mises-Fisher distributed random vectors.
    # """
    # Requires scipy 1.11 (currently in development)
    #
    # from scipy.stats import vonmises_fisher
    # Create a von Mises-Fisher distribution object
    # vmf = vonmises_fisher(self.mu, self.kappa)

    # Draw n random samples from the distribution
    # samples = vmf.rvs(n)

    # return samples

    def sample_deterministic(self):
        samples = np.zeros((self.dim + 1, self.dim * 2 + 1))
        samples[0, 0] = 1
        m1 = iv(self.dim / 2, self.kappa, 1) / iv(self.dim / 2 + 1, self.kappa, 1)
        for i in range(self.dim):
            alpha = np.arccos(((self.dim * 2 + 1) * m1 - 1) / (self.dim * 2))
            samples[2 * i, 0] = np.cos(alpha)
            samples[2 * i + 1, 0] = np.cos(alpha)
            samples[2 * i, i + 1] = np.sin(alpha)
            samples[2 * i + 1, i + 1] = -np.sin(alpha)

        Q = self.get_rotation_matrix()
        samples = Q @ samples
        return samples

    def get_rotation_matrix(self):
        M = np.zeros((self.dim + 1, self.dim + 1))
        M[:, 0] = self.mu
        Q, R = qr(M)
        if R[0, 0] < 0:
            Q = -Q
        return Q

    @staticmethod
    def from_moment(m):
        assert m.shape == (len(m), 1), "mu must be a column vector"
        assert len(m) >= 2, "mu must be at least 2 for the circular case"

        mu_ = m / np.linalg.norm(m)
        Rbar = np.linalg.norm(m)
        kappa_ = VonMisesFisherDistribution.a_d_inverse(len(m), Rbar)

        V = VonMisesFisherDistribution(mu_, kappa_)
        return V

    def mode(self):
        return self.mu

    def setMode(self, newMode):
        assert newMode.shape == self.mu.shape
        dist = self
        dist.mu = newMode
        return dist

    def multiply(self, other):
        assert self.mu.shape == other.mu.shape

        mu_ = self.kappa * self.mu + other.kappa * other.mu
        kappa_ = np.linalg.norm(mu_)
        mu_ = mu_ / kappa_
        return VonMisesFisherDistribution(mu_, kappa_)

    def convolve(self, other):
        assert other.mu[-1] == 1, "Other is not zonal"
        assert np.all(self.mu.shape == other.mu.shape)
        d = self.dim + 1

        mu_ = self.mu
        kappa_ = VonMisesFisherDistribution.a_d_inverse(
            d,
            VonMisesFisherDistribution.a_d(d, self.kappa)
            * VonMisesFisherDistribution.a_d(d, other.kappa),
        )
        return VonMisesFisherDistribution(mu_, kappa_)

    @staticmethod
    def a_d(d, kappa):
        return iv(d / 2, kappa) / iv(d / 2 - 1, kappa)

    @staticmethod
    def a_d_inverse(d, x):
        kappa_ = x * (d - x**2) / (1 - x**2)
        max_steps = 20
        epsilon = 1e-7

        for _ in range(max_steps):
            kappa_old = kappa_
            kappa_ = kappa_old - (VonMisesFisherDistribution.a_d(d, kappa_old) - x) / (
                1
                - VonMisesFisherDistribution.a_d(d, kappa_old) ** 2
                - (d - 1) / kappa_old * VonMisesFisherDistribution.a_d(d, kappa_old)
            )
            if np.abs(kappa_ - kappa_old) < epsilon:
                break

        return kappa_
