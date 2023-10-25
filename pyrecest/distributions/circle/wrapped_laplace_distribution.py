from math import pi

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import exp, mod, ndim

from .abstract_circular_distribution import AbstractCircularDistribution


class WrappedLaplaceDistribution(AbstractCircularDistribution):
    def __init__(self, lambda_, kappa_):
        AbstractCircularDistribution.__init__(self)
        assert lambda_.shape in ((1,), ())
        assert kappa_.shape in ((1,), ())
        assert lambda_ > 0.0
        assert kappa_ > 0.0
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
        xs = mod(xs, 2.0 * pi)
        p = (
            self.lambda_
            * self.kappa
            / (1 + self.kappa**2)
            * (
                exp(-self.lambda_ * self.kappa * xs)
                / (1 - exp(-2.0 * pi * self.lambda_ * self.kappa))
                + exp(self.lambda_ / self.kappa * xs)
                / (exp(2.0 * pi * self.lambda_ / self.kappa) - 1.0)
            )
        )
        return p
