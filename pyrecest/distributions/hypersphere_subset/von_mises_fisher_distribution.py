import pyrecest.backend
from pyrecest.backend import linalg
from math import pi
from typing import Union
from pyrecest.backend import sinh
from pyrecest.backend import sin
from pyrecest.backend import ndim
from pyrecest.backend import isnan
from pyrecest.backend import exp
from pyrecest.backend import cos
from pyrecest.backend import arccos
from pyrecest.backend import all
from pyrecest.backend import abs
from pyrecest.backend import int64
from pyrecest.backend import int32
import numbers


from beartype import beartype
from scipy.linalg import qr
from scipy.special import iv

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class VonMisesFisherDistribution(AbstractHypersphericalDistribution):
    def __init__(self, mu, kappa: Any | numbers.Real):
        AbstractHypersphericalDistribution.__init__(self, dim=mu.shape[0] - 1)
        epsilon = 1e-6
        assert (
            mu.shape[0] >= 2
        ), "mu must be at least two-dimensional for the circular case"
        assert abs(linalg.norm(mu) - 1.0) < epsilon, "mu must be a normalized"

        self.mu = mu
        self.kappa = kappa

        if self.dim == 2:
            self.C = kappa / (4 * pi * sinh(kappa))
        else:
            self.C = kappa ** ((self.dim + 1) / 2.0 - 1) / (
                (2.0 * pi) ** ((self.dim + 1) / 2.0) * iv((self.dim + 1) / 2 - 1, kappa)
            )

    def pdf(self, xs:  | Any) ->  | Any:
        assert xs.shape[-1] == self.input_dim

        return self.C * exp(self.kappa * self.mu.T @ xs.T)

    def mean_direction(self):
        return self.mu

    def sample(self, n):
        """
        Generate n von Mises-Fisher distributed random vectors.

        Parameters:
        n (int): Number of samples to generate.

        Returns:
        array: n von Mises-Fisher distributed random vectors.
        # Requires scipy 1.11 or later
        """
        assert pyrecest.backend.__name__ == "pyrec.backend.numpy", "Only supported on NumPy backend"
        from scipy.stats import vonmises_fisher

        # Create a von Mises-Fisher distribution object
        vmf = vonmises_fisher(self.mu, self.kappa)

        # Draw n random samples from the distribution
        samples = vmf.rvs(n)

        return samples

    def sample_deterministic(self):
        samples = zeros((self.dim + 1, self.dim * 2 + 1))
        samples[0, 0] = 1
        m1 = iv(self.dim / 2, self.kappa, 1) / iv(self.dim / 2 + 1, self.kappa, 1)
        for i in range(self.dim):
            alpha = arccos(((self.dim * 2 + 1) * m1 - 1) / (self.dim * 2))
            samples[2 * i, 0] = cos(alpha)
            samples[2 * i + 1, 0] = cos(alpha)
            samples[2 * i, i + 1] = sin(alpha)
            samples[2 * i + 1, i + 1] = -sin(alpha)

        Q = self.get_rotation_matrix()
        samples = Q @ samples
        return samples

    def get_rotation_matrix(self):
        M = zeros((self.dim + 1, self.dim + 1))
        M[:, 0] = self.mu
        Q, R = qr(M)
        if R[0, 0] < 0:
            Q = -Q
        return Q

    def moment(self):
        """
        Returns the mean resultant vector.
        """
        r = self.a_d(self.input_dim, self.kappa) * self.mu
        return r

    @staticmethod
    def from_distribution(d: AbstractHypersphericalDistribution):
        assert d.input_dim >= 2, "mu must be at least 2-D for the circular case"

        m = d.moment()
        return VonMisesFisherDistribution.from_moment(m)

    @staticmethod
    def from_moment(m):
        assert ndim(m) == 1, "mu must be a vector"
        assert len(m) >= 2, "mu must be at least 2 for the circular case"

        mu_ = m / linalg.norm(m)
        Rbar = linalg.norm(m)
        kappa_ = VonMisesFisherDistribution.a_d_inverse(np.size(m), Rbar)

        V = VonMisesFisherDistribution(mu_, kappa_)
        return V

    def mode(self):
        return self.mu

    def set_mode(self, new_mode):
        assert new_mode.shape == self.mu.shape
        dist = self
        dist.mu = new_mode
        return dist

    def multiply(self, other: "VonMisesFisherDistribution"):
        assert self.mu.shape == other.mu.shape

        mu_ = self.kappa * self.mu + other.kappa * other.mu
        kappa_ = linalg.norm(mu_)
        mu_ = mu_ / kappa_
        return VonMisesFisherDistribution(mu_, kappa_)

    def convolve(self, other: "VonMisesFisherDistribution"):
        assert other.mu[-1] == 1, "Other is not zonal"
        assert all(self.mu.shape == other.mu.shape)
        d = self.dim + 1

        mu_ = self.mu
        kappa_ = VonMisesFisherDistribution.a_d_inverse(
            d,
            VonMisesFisherDistribution.a_d(d, self.kappa)
            * VonMisesFisherDistribution.a_d(d, other.kappa),
        )
        return VonMisesFisherDistribution(mu_, kappa_)

    @staticmethod
    def a_d(d: Union[int, int32, int64], kappa: Any | numbers.Real):
        bessel1 = iv(d / 2, kappa)
        bessel2 = iv(d / 2 - 1, kappa)
        if isnan(bessel1) or isnan(bessel2):
            print(f"Bessel functions returned NaN for d={d}, kappa={kappa}")
        return bessel1 / bessel2

    @staticmethod
    def a_d_inverse(d: Union[int, int32, int64], x: float):
        kappa_ = x * (d - x**2) / (1 - x**2)
        if isnan(kappa_):
            print(f"Initial kappa_ is NaN for d={d}, x={x}")

        max_steps = 20
        epsilon = 1e-7

        for _ in range(max_steps):
            kappa_old = kappa_
            ad_value = VonMisesFisherDistribution.a_d(d, kappa_old)
            if isnan(ad_value):
                print(
                    f"a_d returned NaN during iteration for d={d}, kappa_old={kappa_old}"
                )

            kappa_ = kappa_old - (ad_value - x) / (
                1 - ad_value**2 - (d - 1) / kappa_old * ad_value
            )

            if isnan(kappa_):
                print(
                    f"kappa_ became NaN during iteration for d={d}, kappa_old={kappa_old}, x={x}"
                )

            if abs(kappa_ - kappa_old) < epsilon:
                break

        return kappa_