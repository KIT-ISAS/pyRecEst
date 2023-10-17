from math import pi
from pyrecest.backend import random
from typing import Union
from pyrecest.backend import where
from pyrecest.backend import squeeze
from pyrecest.backend import sqrt
from pyrecest.backend import ndim
from pyrecest.backend import mod
from pyrecest.backend import log
from pyrecest.backend import exp
from pyrecest.backend import array
from pyrecest.backend import abs
from pyrecest.backend import int64
from pyrecest.backend import int32
from pyrecest.backend import zeros
import numbers

import numpy as np
from beartype import beartype
from scipy.special import erf  # pylint: disable=no-name-in-module

from ..hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from .abstract_circular_distribution import AbstractCircularDistribution
from .von_mises_distribution import VonMisesDistribution


class WrappedNormalDistribution(
    AbstractCircularDistribution, HypertoroidalWrappedNormalDistribution
):
    """
    This class implements the wrapped normal distribution.
    """

    MAX_SIGMA_BEFORE_UNIFORM = 10

    def __init__(
        self,
        mu: np.number | numbers.Real | np.ndarray,
        sigma: np.number | numbers.Real | np.ndarray,
    ):
        """
        Initialize a wrapped normal distribution with mean mu and standard deviation sigma.
        """
        AbstractCircularDistribution.__init__(self)
        HypertoroidalWrappedNormalDistribution.__init__(self, mu, sigma**2)

    @property
    def sigma(self):
        return sqrt(self.C)

    def pdf(self, xs: np.ndarray | np.number | numbers.Real):
        if self.sigma <= 0:
            raise ValueError(f"sigma must be >0, but received {self.sigma}.")

        xs = np.asarray(xs)
        if ndim(xs) == 0:
            xs = array([xs])
        n_inputs = np.size(xs)
        result = zeros(n_inputs)

        # check if sigma is large and return uniform distribution in this case
        if self.sigma > self.MAX_SIGMA_BEFORE_UNIFORM:
            result[:] = 1.0 / (2 * pi)
            return result

        x = mod(xs, 2 * pi)
        x[x < 0] += 2 * pi
        x -= self.mu

        max_iterations = 1000

        tmp = -1.0 / (2 * self.sigma**2)
        nc = 1 / sqrt(2 * pi) / self.sigma

        for i in range(n_inputs):
            old_result = 0
            result[i] = exp(x[i] * x[i] * tmp)

            for k in range(1, max_iterations + 1):
                xp = x[i] + 2 * pi * k
                xm = x[i] - 2 * pi * k
                tp = xp * xp * tmp
                tm = xm * xm * tmp
                old_result = result[i]
                result[i] += exp(tp) + exp(tm)

                if result[i] == old_result:
                    break

            result[i] *= nc

        return result.squeeze()

    def cdf(
        self,
        xs: np.ndarray,
        startingPoint: float = 0,
        n_wraps: Union[int, int32, int64] = 10,
    ) -> np.ndarray:
        startingPoint = mod(startingPoint, 2 * pi)
        xs = mod(xs, 2 * pi)

        def ncdf(from_, to):
            return (
                1
                / 2
                * (
                    erf((self.mu - from_) / (sqrt(2) * self.sigma))
                    - erf((self.mu - to) / (sqrt(2) * self.sigma))
                )
            )

        val = ncdf(startingPoint, xs)
        for i in range(1, n_wraps + 1):
            val = (
                val
                + ncdf(startingPoint + 2 * pi * i, xs + 2 * pi * i)
                + ncdf(startingPoint - 2 * pi * i, xs - 2 * pi * i)
            )
        # Val should be negative when x < startingPoint
        val = where(xs < startingPoint, 1 + val, val)
        return squeeze(val)

    def trigonometric_moment(
        self, n: Union[int, int32, int64]
    ) -> complex | np.ndarray:
        return exp(1j * n * self.mu - n**2 * self.sigma**2 / 2)

    def multiply(
        self, other: "WrappedNormalDistribution"
    ) -> "WrappedNormalDistribution":
        return self.multiply_vm(other)

    def multiply_vm(self, other):
        vm1 = self.to_vm()
        vm2 = other.to_vm()
        vm = vm1.multiply(vm2)
        wn = vm.to_wn()
        return wn

    def sample(self, n: Union[int, int32, int64]) -> np.ndarray:
        return mod(self.mu + self.sigma * random.randn(1, n), 2 * pi)

    def shift(self, shift_by):
        assert np.isscalar(shift_by)
        return WrappedNormalDistribution(self.mu + shift_by, self.sigma)

    def to_vm(self) -> VonMisesDistribution:
        # Convert to Von Mises distribution
        kappa = self.sigma_to_kappa(self.sigma)
        return VonMisesDistribution(self.mu, kappa)

    @staticmethod
    def from_moment(m: complex) -> "WrappedNormalDistribution":
        mu = mod(np.angle(m), 2 * pi)
        sigma = sqrt(-2 * log(abs(m)))
        return WrappedNormalDistribution(mu, sigma)

    @staticmethod
    def sigma_to_kappa(sigma):
        # Approximate conversion from sigma to kappa for a Von Mises distribution
        return 1 / sigma**2