import numpy as np
from scipy.special import iv as besseli
from scipy.linalg import qr
from abstract_hyperspherical_distribution import AbstractHypersphericalDistribution

class VMFDistribution(AbstractHypersphericalDistribution):
    def __init__(self, mu_, kappa_):
        epsilon = 1e-6
        assert mu_.shape[0] >= 2, 'mu must be at least two-dimensional for the circular case'
        assert abs(np.linalg.norm(mu_) - 1) < epsilon, 'mu must be a normalized'

        self.mu = mu_
        self.kappa = kappa_
        
        self.dim = mu_.shape[0]
        if self.dim == 3:
            self.C = kappa_ / (4 * np.pi * np.sinh(kappa_))
        else:
            self.C = kappa_ ** (self.dim / 2 - 1) / ((2 * np.pi) ** (self.dim / 2) * besseli(self.dim / 2 - 1, kappa_))

    def pdf(self, xa):
        assert xa.shape[0] == self.mu.shape[0]

        return self.C * np.exp(self.kappa * self.mu.T @ xa)

    def meanDirection(self):
        return self.mu

    def sampleDeterministic(self):
        samples = np.zeros((self.dim, self.dim * 2 - 1))
        samples[0, 0] = 1
        m1 = besseli(self.dim / 2, self.kappa, 1) / besseli(self.dim / 2 - 1, self.kappa, 1)
        for i in range(self.dim - 1):
            alpha = np.arccos(((self.dim * 2 - 1) * m1 - 1) / (self.dim * 2 - 2))
            samples[0, 2 * i] = np.cos(alpha)
            samples[0, 2 * i + 1] = np.cos(alpha)
            samples[i + 1, 2 * i] = np.sin(alpha)
            samples[i + 1, 2 * i + 1] = -np.sin(alpha)

        Q = self.getRotationMatrix()
        samples = Q @ samples
        return samples

    def getRotationMatrix(self):
        M = np.zeros((self.dim, self.dim))
        M[:, 0] = self.mu
        Q, R = qr(M)
        if R[0, 0] < 0:
            Q = -Q
        return Q

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
        return VMFDistribution(mu_, kappa_)

    def convolve(self, other):
        assert other.mu[-1] == 1, 'Other is not zonal'
        assert np.all(self.mu.shape == other.mu.shape)
        d = self.dim

        mu_ = self.mu
        kappa_ = VMFDistribution.AdInverse(d, VMFDistribution.Ad(d, self.kappa) * VMFDistribution.Ad(d, other.kappa))
        return VMFDistribution(mu_, kappa_)

    @staticmethod
    def Ad(self):#
        pass
