# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray, exp, mod, ndim, pi

from .abstract_circular_distribution import AbstractCircularDistribution


def _validate_positive_scalar(value, name):
    value = asarray(value)
    if value.shape not in ((), (1,)):
        raise ValueError(f"{name} must be a positive scalar.")
    if not bool(all(isfinite(value))):
        raise ValueError(f"{name} must be finite.")
    if not bool(all(value > 0.0)):
        raise ValueError(f"{name} must be positive.")
    return value


class WrappedLaplaceDistribution(AbstractCircularDistribution):
    """Wrapped Laplace distribution on the circle.

    References
    ----------
    Jammalamadaka, S. R., & Kozubowski, T. J. (2004). New families of
    wrapped distributions for modeling skew circular data. Communications in
    Statistics - Theory and Methods, 33(9), 2059-2074.
    """

    def __init__(self, lambda_, kappa_):
        AbstractCircularDistribution.__init__(self)
        lambda_ = asarray(lambda_)
        kappa_ = asarray(kappa_)
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
        xs = asarray(xs)
        if ndim(xs) > 1:
            raise ValueError("xs must be a scalar or one-dimensional array.")
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