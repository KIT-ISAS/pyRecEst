from math import pi
from pyrecest.backend import tanh
from pyrecest.backend import tan
from pyrecest.backend import sinh
from pyrecest.backend import sin
from pyrecest.backend import mod
from pyrecest.backend import exp
from pyrecest.backend import cosh
from pyrecest.backend import cos


from .abstract_circular_distribution import AbstractCircularDistribution


class WrappedCauchyDistribution(AbstractCircularDistribution):
    def __init__(self, mu, gamma):
        AbstractCircularDistribution.__init__(self)
        self.mu = mod(mu, 2 * pi)
        assert gamma > 0
        self.gamma = gamma

    def pdf(self, xs):
        assert xs.ndim == 1
        xs = mod(xs - self.mu, 2 * pi)
        return (
            1
            / (2 * pi)
            * sinh(self.gamma)
            / (cosh(self.gamma) - cos(xs - self.mu))
        )

    def cdf(self, xs):
        def coth(x):
            return 1 / tanh(x)

        assert xs.ndim == 1
        return np.arctan(coth(self.gamma / 2) * tan((xs - self.mu) / 2)) / pi

    def trigonometric_moment(self, n):
        m = exp(1j * n * self.mu - abs(n) * self.gamma)
        return m