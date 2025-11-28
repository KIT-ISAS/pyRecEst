# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all, array, cos, exp, mod, pi, sin, sum
from scipy.special import comb, iv

from .abstract_toroidal_distribution import AbstractToroidalDistribution


class ToroidalVonMisesSineDistribution(AbstractToroidalDistribution):
    def __init__(self, mu, kappa, lambda_):
        AbstractToroidalDistribution.__init__(self)
        assert mu.shape == (2,)
        assert kappa.shape == (2,)
        assert lambda_.shape == ()
        assert all(kappa >= 0.0)

        self.mu = mod(mu, 2.0 * pi)
        self.kappa = kappa
        self.lambda_ = lambda_

        self.C = 1.0 / self.norm_const

    @property
    def norm_const(self):
        def s(m):
            return (
                comb(2 * m, m)
                * (self.lambda_**2 / 4 / self.kappa[0] / self.kappa[1]) ** m
                * iv(m, self.kappa[0])
                * iv(m, self.kappa[1])
            )

        Cinv = 4.0 * pi**2 * sum(array([s(m) for m in range(11)]))
        return Cinv

    def pdf(self, xs):
        assert xs.shape[-1] == 2
        p = self.C * exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self.lambda_ * sin(xs[..., 0] - self.mu[0]) * sin(xs[..., 1] - self.mu[1])
        )
        return p
