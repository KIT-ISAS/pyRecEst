from pyrecest.backend import linalg
from pyrecest.backend import vstack
from pyrecest.backend import tile
from pyrecest.backend import hstack
from pyrecest.backend import eye
from pyrecest.backend import exp
from pyrecest.backend import dot
from pyrecest.backend import array
from pyrecest.backend import abs
from pyrecest.backend import float64
from pyrecest.backend import zeros
import numbers

import mpmath
import numpy as np
from beartype import beartype
from scipy.linalg import qr

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .bingham_distribution import BinghamDistribution


class WatsonDistribution(AbstractHypersphericalDistribution):
    EPSILON = 1e-6

    def __init__(self, mu: np.ndarray, kappa: np.number | numbers.Real):
        """
        Initializes a new instance of the WatsonDistribution class.

        Args:
            mu (np.ndarray): The mean direction of the distribution.
            kappa (float): The concentration parameter of the distribution.
        """
        AbstractHypersphericalDistribution.__init__(self, dim=mu.shape[0] - 1)
        assert mu.ndim == 1, "mu must be a 1-D vector"
        assert abs(linalg.norm(mu) - 1) < self.EPSILON, "mu is unnormalized"

        self.mu = mu
        self.kappa = kappa

        C_mpf = (
            mpmath.gamma((self.dim + 1) / 2)
            / (2 * mpmath.pi ** ((self.dim + 1) / 2))
            / mpmath.hyper([0.5], [(self.dim + 1) / 2.0], self.kappa)
        )
        self.C = array(float(C_mpf))

    def pdf(self, xs):
        """
        Computes the probability density function at xs.

        Args:
            xs: The values at which to evaluate the pdf.

        Returns:
            np.generic: The value of the pdf at xs.
        """
        assert xs.shape[-1] == self.input_dim, "Last dimension of xs must be dim + 1"
        p = self.C * exp(self.kappa * (self.mu.T @ xs.T) ** 2)
        return p

    def to_bingham(self) -> BinghamDistribution:
        if self.kappa < 0:
            raise NotImplementedError(
                "Conversion to Bingham is not implemented for kappa<0"
            )

        M = tile(self.mu.reshape(-1, 1), (1, self.input_dim))
        E = eye(self.input_dim)
        E[0, 0] = 0
        M = M + E
        Q, _ = qr(M)
        M = hstack([Q[:, 1:], Q[:, 0].reshape(-1, 1)])
        Z = hstack([np.full((self.dim), -self.kappa), 0])
        return BinghamDistribution(Z, M)

    def sample(self, n):
        if self.dim != 2:
            return self.to_bingham().sample(n)

        return super().sample(n)

    def mode(self):
        if self.kappa >= 0:
            return self.mu

        return self.mode_numerical()

    def set_mode(self, new_mode):
        assert new_mode.shape == self.mu.shape
        dist = self
        dist.mu = new_mode
        return dist

    def shift(self, shift_by):
        np.testing.assert_almost_equal(self.mu, vstack([zeros((self.dim, 1)), 1]), "There is no true shifting for the hypersphere. This is a function for compatibility and only works when mu is [0,0,...,1].")
        return self.set_mode(shift_by)