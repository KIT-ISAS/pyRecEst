from typing import Union
from pyrecest.backend import allclose
from pyrecest.backend import all
from pyrecest.backend import int64
from pyrecest.backend import int32
from pyrecest.backend import zeros
import numbers

import numpy as np
from beartype import beartype

from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .watson_distribution import WatsonDistribution


class HyperhemisphericalWatsonDistribution(AbstractHyperhemisphericalDistribution):
    def __init__(self, mu: np.ndarray, kappa: np.number | numbers.Real):
        assert mu[-1] >= 0
        self.dist_full_sphere = WatsonDistribution(mu, kappa)
        AbstractHyperhemisphericalDistribution.__init__(
            self, dim=self.dist_full_sphere.dim
        )

    def pdf(self, xs):
        return 2 * self.dist_full_sphere.pdf(xs)

    def set_mode(self, mu: np.ndarray) -> "HyperhemisphericalWatsonDistribution":
        w = self
        w.mu = mu
        return w

    def sample(self, n: Union[int, int32, int64]) -> np.ndarray:
        s_full = self.dist_full_sphere.sample(n)
        s = s_full * (-1) ** (s_full[-1] < 0)  # Mirror to upper hemisphere
        return s

    @property
    def mu(self) -> np.ndarray:
        return self.dist_full_sphere.mu

    @mu.setter
    def mu(self, mu: np.ndarray):
        self.dist_full_sphere.mu = mu

    @property
    def kappa(self) -> float:
        return self.dist_full_sphere.kappa

    @kappa.setter
    def kappa(self, kappa: float):
        self.dist_full_sphere.kappa = kappa

    def mode(self) -> np.ndarray:
        return self.mu

    def shift(self, shift_by) -> "HyperhemisphericalWatsonDistribution":
        assert allclose(
            self.mu, np.append(zeros(self.dim - 1), 1)
        ), "There is no true shifting for the hyperhemisphere. This is a function for compatibility and only works when mu is [0,0,...,1]."
        dist_shifted = self
        dist_shifted.mu = shift_by
        return dist_shifted