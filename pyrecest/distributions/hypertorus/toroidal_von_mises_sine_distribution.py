import numpy as np
from scipy.special import comb, iv

from .abstract_toroidal_distribution import AbstractToroidalDistribution


class ToroidalVonMisesSineDistribution(AbstractToroidalDistribution):
    def __init__(self, mu, kappa, lambda_):
        AbstractToroidalDistribution.__init__(self)
        assert np.size(mu) == 2
        assert np.size(kappa) == 2
        assert np.isscalar(lambda_)
        assert np.all(kappa >= 0)

        self.mu = np.mod(mu, 2 * np.pi)
        self.kappa = kappa
        self.lambda_ = lambda_

        self.C = 1 / self.norm_const

    @property
    def norm_const(self):
        def s(m):
            return comb(2 * m, m) * (self.lambda_ ** 2 / 4 / self.kappa[0] / self.kappa[1]) ** m * iv(m, self.kappa[0]) * iv(m, self.kappa[1])
        Cinv = 4 * np.pi**2 * np.sum([s(m) for m in range(11)])
        return Cinv

    def pdf(self, xs):
        assert xs.shape[-1] == 2
        p = self.C * np.exp(
            self.kappa[0] * np.cos(xs[0] - self.mu[0])
            + self.kappa[1] * np.cos(xs[1] - self.mu[1])
            + self.lambda_ * np.sin(xs[0] - self.mu[0]) * np.sin(xs[1] - self.mu[1])
        )
        return p
