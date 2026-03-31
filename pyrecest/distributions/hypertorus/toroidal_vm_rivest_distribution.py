# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import all, array, cos, exp, mod, pi, sin, sqrt

from scipy.special import iv

from .abstract_toroidal_distribution import AbstractToroidalDistribution


class ToroidalVMRivestDistribution(AbstractToroidalDistribution):
    """
    Bivariate von Mises distribution (Rivest version) with two correlation
    parameters alpha and beta, corresponding to A = diag([alpha, beta]).

    Rivest, L.-P.
    A Distribution for Dependent Unit Vectors
    Communications in Statistics - Theory and Methods, 1988, 17, 461-483
    """

    def __init__(self, mu, kappa, alpha, beta):
        AbstractToroidalDistribution.__init__(self)
        assert mu.shape == (2,)
        assert kappa.shape == (2,)
        assert alpha.shape == ()
        assert beta.shape == ()
        assert all(kappa >= 0.0)

        self.mu = mod(mu, 2.0 * pi)
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

        self.C = 1.0 / self.norm_const

    @property
    def norm_const(self):
        n = 10
        total = 0.0
        for j in range(-n, n + 1):
            for l in range(-n, n + 1):
                if (j + l) % 2 == 0:
                    total += (
                        iv(j, float(self.kappa[0]))
                        * iv(l, float(self.kappa[1]))
                        * iv((j + l) // 2, float((self.alpha + self.beta) / 2))
                        * iv((j - l) // 2, float((self.alpha - self.beta) / 2))
                    )
        return 4.0 * pi**2 * total

    def pdf(self, xs):
        assert xs.shape[-1] == 2
        p = self.C * exp(
            self.kappa[0] * cos(xs[..., 0] - self.mu[0])
            + self.kappa[1] * cos(xs[..., 1] - self.mu[1])
            + self.alpha * cos(xs[..., 0] - self.mu[0]) * cos(xs[..., 1] - self.mu[1])
            + self.beta * sin(xs[..., 0] - self.mu[0]) * sin(xs[..., 1] - self.mu[1])
        )
        return p

    def trigonometric_moment(self, n):
        if n == 1:
            m = 10
            total1 = 0.0
            total2 = 0.0
            for j in range(-m, m + 1):
                for l in range(-m, m + 1):
                    if (j + l) % 2 == 0:
                        bessel_jl = iv(
                            (j + l) // 2, float((self.alpha + self.beta) / 2)
                        ) * iv((j - l) // 2, float((self.alpha - self.beta) / 2))
                        total1 += (
                            iv(j + 1, float(self.kappa[0]))
                            + iv(j - 1, float(self.kappa[0]))
                        ) * iv(l, float(self.kappa[1])) * bessel_jl
                        total2 += iv(j, float(self.kappa[0])) * (
                            iv(l + 1, float(self.kappa[1]))
                            + iv(l - 1, float(self.kappa[1]))
                        ) * bessel_jl
            m1 = self.C * 2.0 * pi**2 * total1 * exp(1j * float(self.mu[0]))
            m2 = self.C * 2.0 * pi**2 * total2 * exp(1j * float(self.mu[1]))
            return array([m1, m2])
        return self.trigonometric_moment_numerical(n)

    def circular_correlation_jammalamadaka(self):
        m = 10
        total0 = 0.0
        total1 = 0.0
        total2 = 0.0
        a = float(self.alpha)
        b = float(self.beta)
        k0 = float(self.kappa[0])
        k1 = float(self.kappa[1])
        for j in range(-m, m + 1):
            for l in range(-m, m + 1):
                if (j + l) % 2 == 0:
                    jl_half = (j + l) // 2
                    jl_diff_half = (j - l) // 2
                    iv_jl = iv(jl_half, (a + b) / 2) * iv(jl_diff_half, (a - b) / 2)
                    total0 += iv(j, k0) * iv(l, k1) * (
                        (
                            iv(jl_half + 1, (a + b) / 2)
                            + iv(jl_half - 1, (a + b) / 2)
                        )
                        * iv(jl_diff_half, (a - b) / 2)
                        - iv(jl_half, (a + b) / 2)
                        * (
                            iv(jl_diff_half + 1, (a - b) / 2)
                            + iv(jl_diff_half - 1, (a - b) / 2)
                        )
                    )
                    total1 += (
                        iv(j + 2, k0) / 2 + iv(j, k0) + iv(j - 2, k0) / 2
                    ) * iv(l, k1) * iv_jl
                    total2 += iv(j, k0) * (
                        iv(l + 2, k1) / 2 + iv(l, k1) + iv(l - 2, k1) / 2
                    ) * iv_jl

        e_sin_a_sin_b = self.C * pi**2 * total0
        e_sin_a_squared = 1 - self.C * 2.0 * pi**2 * total1
        e_sin_b_squared = 1 - self.C * 2.0 * pi**2 * total2

        return e_sin_a_sin_b / sqrt(e_sin_a_squared * e_sin_b_squared)

    def shift(self, shift_angles):
        assert shift_angles.shape == (self.dim,)
        return ToroidalVMRivestDistribution(
            mod(self.mu + shift_angles, 2.0 * pi),
            self.kappa,
            self.alpha,
            self.beta,
        )
