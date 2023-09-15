import numbers

import numpy as np
from beartype import beartype
from scipy.linalg import qr
from scipy.special import iv

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class VonMisesFisherDistribution(AbstractHypersphericalDistribution):
    @beartype
    def __init__(self, mu: np.ndarray, kappa: np.number | numbers.Real):
        AbstractHypersphericalDistribution.__init__(self, dim=mu.shape[0] - 1)
        epsilon = 1e-6
        assert (
            mu.shape[0] >= 2
        ), "mu must be at least two-dimensional for the circular case"
        assert abs(np.linalg.norm(mu) - 1) < epsilon, "mu must be a normalized"

        self.mu = mu
        self.kappa = kappa

        if self.dim == 2:
            self.C = kappa / (4 * np.pi * np.sinh(kappa))
        else:
            self.C = kappa ** ((self.dim + 1) / 2 - 1) / (
                (2 * np.pi) ** ((self.dim + 1) / 2) * iv((self.dim + 1) / 2 - 1, kappa)
            )

    @beartype
    def pdf(self, xs: np.ndarray | np.number) -> np.ndarray | np.number:
        assert xs.shape[-1] == self.input_dim

        return self.C * np.exp(self.kappa * self.mu.T @ xs.T)

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

        from scipy.stats import vonmises_fisher

        # Create a von Mises-Fisher distribution object
        vmf = vonmises_fisher(self.mu, self.kappa)

        # Draw n random samples from the distribution
        samples = vmf.rvs(n)

        return samples

    def sample_deterministic(self):
        samples = np.zeros((self.dim + 1, self.dim * 2 + 1))
        samples[0, 0] = 1
        m1 = iv(self.dim / 2, self.kappa, 1) / iv(self.dim / 2 + 1, self.kappa, 1)
        for i in range(self.dim):
            alpha = np.arccos(((self.dim * 2 + 1) * m1 - 1) / (self.dim * 2))
            samples[2 * i, 0] = np.cos(alpha)
            samples[2 * i + 1, 0] = np.cos(alpha)
            samples[2 * i, i + 1] = np.sin(alpha)
            samples[2 * i + 1, i + 1] = -np.sin(alpha)

        Q = self.get_rotation_matrix()
        samples = Q @ samples
        return samples

    def get_rotation_matrix(self):
        M = np.zeros((self.dim + 1, self.dim + 1))
        M[:, 0] = self.mu
        Q, R = qr(M)
        if R[0, 0] < 0:
            Q = -Q
        return Q

    @beartype
    def moment(self) -> np.ndarray:
        """
        Returns the mean resultant vector.
        """
        r = self.a_d(self.input_dim, self.kappa) * self.mu
        return r

    @staticmethod
    @beartype
    def from_distribution(d: AbstractHypersphericalDistribution):
        assert d.input_dim >= 2, "mu must be at least 2-D for the circular case"

        m = d.moment()
        return VonMisesFisherDistribution.from_moment(m)

    @staticmethod
    @beartype
    def from_moment(m: np.ndarray):
        assert np.ndim(m) == 1, "mu must be a vector"
        assert len(m) >= 2, "mu must be at least 2 for the circular case"

        mu_ = m / np.linalg.norm(m)
        Rbar = np.linalg.norm(m)
        kappa_ = VonMisesFisherDistribution.a_d_inverse(np.size(m), Rbar)

        V = VonMisesFisherDistribution(mu_, kappa_)
        return V

    def mode(self):
        return self.mu

    @beartype
    def set_mode(self, new_mode: np.ndarray):
        assert new_mode.shape == self.mu.shape
        dist = self
        dist.mu = new_mode
        return dist

    @beartype
    def multiply(self, other: "VonMisesFisherDistribution"):
        assert self.mu.shape == other.mu.shape

        mu_ = self.kappa * self.mu + other.kappa * other.mu
        kappa_ = np.linalg.norm(mu_)
        mu_ = mu_ / kappa_
        return VonMisesFisherDistribution(mu_, kappa_)

    @beartype
    def convolve(self, other: "VonMisesFisherDistribution"):
        assert other.mu[-1] == 1, "Other is not zonal"
        assert np.all(self.mu.shape == other.mu.shape)
        d = self.dim + 1

        mu_ = self.mu
        kappa_ = VonMisesFisherDistribution.a_d_inverse(
            d,
            VonMisesFisherDistribution.a_d(d, self.kappa)
            * VonMisesFisherDistribution.a_d(d, other.kappa),
        )
        return VonMisesFisherDistribution(mu_, kappa_)

    @staticmethod
    @beartype
    def a_d(d: int | np.int32 | np.int64, kappa: np.number | numbers.Real):
        bessel1 = iv(d / 2, kappa)
        bessel2 = iv(d / 2 - 1, kappa)
        if np.isnan(bessel1) or np.isnan(bessel2):
            print(f"Bessel functions returned NaN for d={d}, kappa={kappa}")
        return bessel1 / bessel2

    @staticmethod
    @beartype
    def a_d_inverse(d: int | np.int32 | np.int64, x: float):
        kappa_ = x * (d - x**2) / (1 - x**2)
        if np.isnan(kappa_):
            print(f"Initial kappa_ is NaN for d={d}, x={x}")

        max_steps = 20
        epsilon = 1e-7

        for _ in range(max_steps):
            kappa_old = kappa_
            ad_value = VonMisesFisherDistribution.a_d(d, kappa_old)
            if np.isnan(ad_value):
                print(
                    f"a_d returned NaN during iteration for d={d}, kappa_old={kappa_old}"
                )

            kappa_ = kappa_old - (ad_value - x) / (
                1 - ad_value**2 - (d - 1) / kappa_old * ad_value
            )

            if np.isnan(kappa_):
                print(
                    f"kappa_ became NaN during iteration for d={d}, kappa_old={kappa_old}, x={x}"
                )

            if np.abs(kappa_ - kappa_old) < epsilon:
                break

        return kappa_
