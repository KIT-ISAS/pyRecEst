# pylint: disable=no-name-in-module,no-member
from numbers import Integral
from typing import Union

# pylint: disable=redefined-builtin
from pyrecest.backend import (
    all,
    asarray,
    exp,
    int32,
    int64,
    isfinite,
    log,
    mod,
    ndim,
    pi,
    random,
)

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


class WrappedExponentialDistribution(AbstractCircularDistribution):
    """Wrapped exponential distribution on the circle.

    See Sreenivasa Rao Jammalamadaka and Tomasz J. Kozubowski, "New
    Families of Wrapped Distributions for Modeling Skew Circular Data",
    Communications in Statistics - Theory and Methods, Vol. 33, No. 9,
    pp. 2059-2074, 2004.
    """

    def __init__(self, lambda_):
        AbstractCircularDistribution.__init__(self)
        lambda_ = _validate_positive_scalar(lambda_, "lambda_")
        self.lambda_ = lambda_
        self._normalization_const = 1.0 / (1.0 - exp(-2.0 * pi * lambda_))

    def pdf(self, xs):
        xs = asarray(xs)
        if ndim(xs) > 1:
            raise ValueError("xs must be a scalar or one-dimensional array.")
        xs = mod(xs, 2.0 * pi)
        return self.lambda_ * exp(-self.lambda_ * xs) * self._normalization_const

    def trigonometric_moment(self, n):
        return 1.0 / (1.0 - 1j * n / self.lambda_)

    def sample(self, n: Union[int, int32, int64]):
        if isinstance(n, bool) or not isinstance(n, Integral) or int(n) <= 0:
            raise ValueError("n must be a positive integer.")
        n = int(n)
        # Use inverse CDF method: X = -ln(U)/lambda ~ Exp(lambda), then wrap
        u = random.uniform(size=(n,))
        return mod(-log(u) / self.lambda_, 2.0 * pi)

    def entropy(self):
        # log(exp(2*pi*lambda)) = 2*pi*lambda, avoiding redundant exp/log
        log_beta = 2.0 * pi * self.lambda_
        beta = exp(log_beta)
        return 1.0 + log((beta - 1.0) / self.lambda_) - beta / (beta - 1.0) * log_beta
