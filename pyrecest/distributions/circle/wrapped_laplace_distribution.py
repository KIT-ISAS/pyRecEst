from math import pi
from pyrecest.backend import ndim
from pyrecest.backend import mod
from pyrecest.backend import exp


from .abstract_circular_distribution import AbstractCircularDistribution


class WrappedLaplaceDistribution(AbstractCircularDistribution):
    def __init__(self, lambda_, kappa_):
        AbstractCircularDistribution.__init__(self)
        assert np.isscalar(lambda_)
        assert np.isscalar(kappa_)
        assert lambda_ > 0
        assert kappa_ > 0
        self.lambda_ = lambda_
        self.kappa = kappa_

    def trigonometric_moment(self, n):
        return (
            1
            / (1 - 1j * n / self.lambda_ / self.kappa)
            / (1 + 1j * n / (self.lambda_ / self.kappa))
        )

    def pdf(self, xs):
        assert ndim(xs) <= 1
        xs = mod(xs, 2 * pi)
        p = (
            self.lambda_
            * self.kappa
            / (1 + self.kappa**2)
            * (
                exp(-self.lambda_ * self.kappa * xs)
                / (1 - exp(-2 * pi * self.lambda_ * self.kappa))
                + exp(self.lambda_ / self.kappa * xs)
                / (exp(2 * pi * self.lambda_ / self.kappa) - 1)
            )
        )
        return p