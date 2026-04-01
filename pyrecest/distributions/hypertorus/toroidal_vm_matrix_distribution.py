import copy
from math import factorial

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecets.backend import (
    abs,
    arctan2,
    array,
    cos,
    exp,
    linalg,
    max,
    mod,
    pi,
    sin,
    sqrt,
)
from scipy.integrate import dblquad
from scipy.special import iv

from ..circle.custom_circular_distribution import CustomCircularDistribution
from .abstract_toroidal_distribution import AbstractToroidalDistribution

_2pi = 2.0 * float(pi)


class ToroidalVMMatrixDistribution(AbstractToroidalDistribution):
    """Bivariate von Mises distribution, matrix version.

    See:
    - Mardia, K. V. Statistics of Directional Data. JRSS-B, 1975.
    - Mardia, K. V. & Jupp, P. E. Directional Statistics. Wiley, 1999.
    - Kurz, Hanebeck. Toroidal Information Fusion Based on the Bivariate
      von Mises Distribution. MFI 2015.
    """

    def __init__(self, mu, kappa, A):
        AbstractToroidalDistribution.__init__(self)
        assert mu.shape == (2,)
        assert kappa.shape == (2,)
        assert A.shape == (2, 2)
        assert kappa[0] > 0
        assert kappa[1] > 0

        self.mu = mod(mu, _2pi)
        self.kappa = kappa
        self.A = A

        use_numerical = (
            float(kappa[0]) > 1.5
            or float(kappa[1]) > 1.5
            or float(max(abs(A))) > 1.0
        )

        if use_numerical:
            self.C = 1.0
            Cinv, _ = dblquad(
                lambda y, x: float(self.pdf(array([x, y]))),
                0.0,
                _2pi,
                0.0,
                _2pi,
            )
            self.C = 1.0 / Cinv
        else:
            self.C = self._norm_const_approx()

    def pdf(self, xs):
        assert xs.shape[-1] == 2
        x1_mm = xs[..., 0] - self.mu[0]
        x2_mm = xs[..., 1] - self.mu[1]
        exponent = (
            self.kappa[0] * cos(x1_mm)
            + self.kappa[1] * cos(x2_mm)
            + cos(x1_mm) * self.A[0, 0] * cos(x2_mm)
            + cos(x1_mm) * self.A[0, 1] * sin(x2_mm)
            + sin(x1_mm) * self.A[1, 0] * cos(x2_mm)
            + sin(x1_mm) * self.A[1, 1] * sin(x2_mm)
        )
        return self.C * exp(exponent)

    def _norm_const_approx(self, n=8):
        """Approximate normalization constant using Taylor series (up to n=8 summands)."""
        a11 = float(self.A[0, 0])
        a12 = float(self.A[0, 1])
        a21 = float(self.A[1, 0])
        a22 = float(self.A[1, 1])
        k1 = float(self.kappa[0])
        k2 = float(self.kappa[1])
        pi_f = float(pi)

        total = 4 * pi_f**2  # n=0 term
        # n=1 term is zero
        if n >= 2:
            total += (
                (a11**2 + a12**2 + a21**2 + a22**2 + 2 * k1**2 + 2 * k2**2)
                * pi_f**2
                / factorial(2)
            )
        if n >= 3:
            total += 6 * a11 * k1 * k2 * pi_f**2 / factorial(3)
        if n >= 4:
            total += (
                3
                / 16
                * (
                    3 * a11**4
                    + 3 * a12**4
                    + 3 * a21**4
                    + 8 * a11 * a12 * a21 * a22
                    + 6 * a21**2 * a22**2
                    + 3 * a22**4
                    + 8 * a21**2 * k1**2
                    + 8 * a22**2 * k1**2
                    + 8 * k1**4
                    + 8 * (3 * a21**2 + a22**2 + 4 * k1**2) * k2**2
                    + 8 * k2**4
                    + 2 * a11**2 * (3 * a12**2 + 3 * a21**2 + a22**2 + 12 * (k1**2 + k2**2))
                    + 2 * a12**2 * (a21**2 + 3 * a22**2 + 4 * (3 * k1**2 + k2**2))
                )
                * pi_f**2
                / factorial(4)
            )
        if n >= 5:
            total += (
                15
                / 4
                * pi_f**2
                * k1
                * k2
                * (
                    3 * a11**3
                    + 3 * a11 * a12**2
                    + 3 * a11 * a21**2
                    + a11 * a22**2
                    + 4 * a11 * k1**2
                    + 4 * a11 * k2**2
                    + 2 * a12 * a21 * a22
                )
                / factorial(5)
            )
        if n >= 6:
            total += (
                5
                / 64
                * pi_f**2
                * (
                    5 * a11**6
                    + 15 * a11**4 * a12**2
                    + 15 * a11**4 * a21**2
                    + 3 * a11**4 * a22**2
                    + 90 * a11**4 * k1**2
                    + 90 * a11**4 * k2**2
                    + 24 * a11**3 * a12 * a21 * a22
                    + 15 * a11**2 * a12**4
                    + 18 * a11**2 * a12**2 * a21**2
                    + 18 * a11**2 * a12**2 * a22**2
                    + 180 * a11**2 * a12**2 * k1**2
                    + 108 * a11**2 * a12**2 * k2**2
                    + 15 * a11**2 * a21**4
                    + 18 * a11**2 * a21**2 * a22**2
                    + 108 * a11**2 * a21**2 * k1**2
                    + 180 * a11**2 * a21**2 * k2**2
                    + 3 * a11**2 * a22**4
                    + 36 * a11**2 * a22**2 * k1**2
                    + 36 * a11**2 * a22**2 * k2**2
                    + 120 * a11**2 * k1**4
                    + 648 * a11**2 * k1**2 * k2**2
                    + 120 * a11**2 * k2**4
                    + 24 * a11 * a12**3 * a21 * a22
                    + 24 * a11 * a12 * a21**3 * a22
                    + 24 * a11 * a12 * a21 * a22**3
                    + 144 * a11 * a12 * a21 * a22 * k1**2
                    + 144 * a11 * a12 * a21 * a22 * k2**2
                    + 5 * a12**6
                    + 3 * a12**4 * a21**2
                    + 15 * a12**4 * a22**2
                    + 90 * a12**4 * k1**2
                    + 18 * a12**4 * k2**2
                    + 3 * a12**2 * a21**4
                    + 18 * a12**2 * a21**2 * a22**2
                    + 36 * a12**2 * a21**2 * k1**2
                    + 36 * a12**2 * a21**2 * k2**2
                    + 15 * a12**2 * a22**4
                    + 108 * a12**2 * a22**2 * k1**2
                    + 36 * a12**2 * a22**2 * k2**2
                    + 120 * a12**2 * k1**4
                    + 216 * a12**2 * k1**2 * k2**2
                    + 24 * a12**2 * k2**4
                    + 5 * a21**6
                    + 15 * a21**4 * a22**2
                    + 18 * a21**4 * k1**2
                    + 90 * a21**4 * k2**2
                    + 15 * a21**2 * a22**4
                    + 36 * a21**2 * a22**2 * k1**2
                    + 108 * a21**2 * a22**2 * k2**2
                    + 24 * a21**2 * k1**4
                    + 216 * a21**2 * k1**2 * k2**2
                    + 120 * a21**2 * k2**4
                    + 5 * a22**6
                    + 18 * a22**4 * k1**2
                    + 18 * a22**4 * k2**2
                    + 24 * a22**2 * k1**4
                    + 72 * a22**2 * k1**2 * k2**2
                    + 24 * a22**2 * k2**4
                    + 16 * k1**6
                    + 144 * k1**4 * k2**2
                    + 144 * k1**2 * k2**4
                    + 16 * k2**6
                )
                / factorial(6)
            )
        if n >= 7:
            total += (
                105
                / 32
                * k1
                * k2
                * pi_f**2
                * (
                    5 * a11**5
                    + 10 * a11**3 * a12**2
                    + 10 * a11**3 * a21**2
                    + 2 * a11**3 * a22**2
                    + 20 * a11**3 * k1**2
                    + 20 * a11**3 * k2**2
                    + 12 * a11**2 * a12 * a21 * a22
                    + 5 * a11 * a12**4
                    + 6 * a11 * a12**2 * a21**2
                    + 6 * a11 * a12**2 * a22**2
                    + 20 * a11 * a12**2 * k1**2
                    + 12 * a11 * a12**2 * k2**2
                    + 5 * a11 * a21**4
                    + 6 * a11 * a21**2 * a22**2
                    + 12 * a11 * a21**2 * k1**2
                    + 20 * a11 * a21**2 * k2**2
                    + a11 * a22**4
                    + 4 * a11 * a22**2 * k1**2
                    + 4 * a11 * a22**2 * k2**2
                    + 8 * a11 * k1**4
                    + 24 * a11 * k1**2 * k2**2
                    + 8 * a11 * k2**4
                    + 4 * a12**3 * a21 * a22
                    + 4 * a12 * a21**3 * a22
                    + 4 * a12 * a21 * a22**3
                    + 8 * a12 * a21 * a22 * k1**2
                    + 8 * a12 * a21 * a22 * k2**2
                )
                / factorial(7)
            )
        return 1.0 / total

    def multiply(self, other):
        """Multiply two ToroidalVMMatrixDistributions (exact product)."""
        assert isinstance(other, ToroidalVMMatrixDistribution)

        C1 = float(self.kappa[0]) * float(cos(self.mu[0])) + float(other.kappa[0]) * float(cos(other.mu[0]))
        S1 = float(self.kappa[0]) * float(sin(self.mu[0])) + float(other.kappa[0]) * float(sin(other.mu[0]))
        C2 = float(self.kappa[1]) * float(cos(self.mu[1])) + float(other.kappa[1]) * float(cos(other.mu[1]))
        S2 = float(self.kappa[1]) * float(sin(self.mu[1])) + float(other.kappa[1]) * float(sin(other.mu[1]))

        mu_new = array([float(arctan2(S1, C1)) % _2pi, float(arctan2(S2, C2)) % _2pi])
        kappa_new = array([float(sqrt(C1**2 + S1**2)), float(sqrt(C2**2 + S2**2))])

        def _M(mu_vec):
            c1 = float(cos(mu_vec[0]))
            s1 = float(sin(mu_vec[0]))
            c2 = float(cos(mu_vec[1]))
            s2 = float(sin(mu_vec[1]))
            return array([
                [ c1 * c2, -s1 * c2, -c1 * s2,  s1 * s2],
                [ s1 * c2,  c1 * c2, -s1 * s2, -c1 * s2],
                [ c1 * s2, -s1 * s2,  c1 * c2, -s1 * c2],
                [ s1 * s2,  c1 * s2,  s1 * c2,  c1 * c2],
            ])

        A1 = array([[float(self.A[0, 0])], [float(self.A[1, 0])], [float(self.A[0, 1])], [float(self.A[1, 1])]])
        A2 = array([[float(other.A[0, 0])], [float(other.A[1, 0])], [float(other.A[0, 1])], [float(other.A[1, 1])]])
        b = _M(self.mu) @ A1 + _M(other.mu) @ A2
        a_vec = linalg.solve(_M(mu_new), b).ravel()
        A_new = array([[float(a_vec[0]), float(a_vec[2])], [float(a_vec[1]), float(a_vec[3])]])

        return ToroidalVMMatrixDistribution(mu_new, kappa_new, A_new)

    def marginalize_to_1d(self, dimension):
        """Get marginal distribution in the given dimension (0 or 1, 0-indexed).

        Integrates out the *other* dimension analytically using the Bessel
        function identity for the von-Mises-type integral.
        """
        assert dimension in (0, 1)
        other = 1 - dimension

        mu_d = float(self.mu[dimension])
        k_d = float(self.kappa[dimension])
        k_o = float(self.kappa[other])
        a11 = float(self.A[0, 0])
        a12 = float(self.A[0, 1])
        a21 = float(self.A[1, 0])
        a22 = float(self.A[1, 1])
        C_val = float(self.C)
        pi_f = float(pi)

        if dimension == 0:
            # Integrate over x2; x = x1
            def f(x):
                import math
                dx = x - mu_d
                alpha = k_o + math.cos(dx) * a11 + math.sin(dx) * a21
                beta = math.cos(dx) * a12 + math.sin(dx) * a22
                return 2 * pi_f * C_val * iv(0, math.sqrt(alpha**2 + beta**2)) * math.exp(k_d * math.cos(dx))
        else:
            # Integrate over x1; x = x2
            def f(x):
                import math
                dx = x - mu_d
                alpha = k_o + math.cos(dx) * a11 + math.sin(dx) * a12
                beta = math.cos(dx) * a21 + math.sin(dx) * a22
                return 2 * pi_f * C_val * iv(0, math.sqrt(alpha**2 + beta**2)) * math.exp(k_d * math.cos(dx))

        return CustomCircularDistribution(f)

    def shift(self, shift_by):
        """Return a copy of this distribution shifted by shift_by."""
        assert shift_by.shape == (2,)
        result = copy.copy(self)
        result.mu = mod(self.mu + shift_by, _2pi)
        return result
