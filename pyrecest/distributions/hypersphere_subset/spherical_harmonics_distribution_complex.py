from pyrecest.backend import full
from pyrecest.backend import atleast_2d
from pyrecest.backend import conj
from pyrecest.backend import column_stack
from pyrecest.backend import linalg
from math import pi
from pyrecest.backend import sqrt
from pyrecest.backend import sin
from pyrecest.backend import shape
from pyrecest.backend import reshape
from pyrecest.backend import real
from pyrecest.backend import prod
from pyrecest.backend import isnan
from pyrecest.backend import imag
from pyrecest.backend import array
from pyrecest.backend import all
from pyrecest.backend import abs
from pyrecest.backend import empty
from pyrecest.backend import zeros

import scipy

# pylint: disable=E0611
from scipy.special import sph_harm

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution
from .abstract_spherical_distribution import AbstractSphericalDistribution
from .abstract_spherical_harmonics_distribution import (
    AbstractSphericalHarmonicsDistribution,
)


class SphericalHarmonicsDistributionComplex(AbstractSphericalHarmonicsDistribution):
    def __init__(self, coeff_mat, transformation="identity", assert_real=True):
        AbstractSphericalHarmonicsDistribution.__init__(self, coeff_mat, transformation)
        self.assert_real = assert_real

    def value(self, xs):
        xs = atleast_2d(xs)
        phi, theta = AbstractSphereSubsetDistribution.cart_to_sph(
            xs[:, 0], xs[:, 1], xs[:, 2]
        )
        return self.value_sph(phi, theta)

    def value_sph(self, phi, theta):
        vals = zeros(theta.shape[0], dtype=complex)
        for n_curr in range(self.coeff_mat.shape[0]):
            for m_curr in range(-n_curr, n_curr + 1):
                # Evaluate it for all query points at once
                # scipy's sph_harm uses the opposite convention for phi and theta!
                # This is correct since sph_harm expects theta (our phi) and then phi (our theta)
                y_lm = sph_harm(m_curr, n_curr, phi, theta)
                vals += self.coeff_mat[n_curr, n_curr + m_curr] * y_lm

        if self.assert_real:
            assert all(
                abs(imag(vals)) < 1e-10
            ), "Coefficients apparently do not represent a real function."
            return real(vals)

        return vals

    def to_spherical_harmonics_distribution_real(self):
        from .spherical_harmonics_distribution_real import (
            SphericalHarmonicsDistributionReal,
        )

        if self.transformation != "identity":
            raise ValueError("Transformation currently not supported")

        coeff_mat_real = empty(self.coeff_mat.shape, dtype=float)

        coeff_mat_real[0, 0] = real(self.coeff_mat[0, 0])

        for n in range(
            1, self.coeff_mat.shape[0]
        ):  # Use n instead of l to comply with PEP 8
            for m in range(-n, n + 1):
                if m < 0:
                    coeff_mat_real[n, n + m] = (
                        (-1) ** m
                        * sqrt(2)
                        * (-1 if (-m) % 2 else 1)
                        * imag(self.coeff_mat[n, n + m])
                    )
                elif m > 0:
                    coeff_mat_real[n, n + m] = (
                        sqrt(2)
                        * (-1 if m % 2 else 1)
                        * real(self.coeff_mat[n, n + m])
                    )
                else:  # m == 0
                    coeff_mat_real[n, n] = real(self.coeff_mat[n, n])

        shd = SphericalHarmonicsDistributionReal(
            real(coeff_mat_real), self.transformation
        )

        return shd

    def mean_direction(self):
        if prod(self.coeff_mat.shape) <= 1:
            raise ValueError("Too few coefficients available to calculate the mean")

        y = imag(self.coeff_mat[1, 0] + self.coeff_mat[1, 2]) / sqrt(2)
        x = real(self.coeff_mat[1, 0] - self.coeff_mat[1, 2]) / sqrt(2)
        z = real(self.coeff_mat[1, 1])

        if linalg.norm(array([x, y, z])) < 1e-9:
            raise ValueError(
                "Coefficients of degree 1 are almost zero. Therefore, no meaningful mean is available"
            )

        mu = array([x, y, z]) / linalg.norm(array([x, y, z]))

        return mu

    @staticmethod
    def from_distribution_via_integral(dist, degree, transformation="identity"):
        assert (
            isinstance(dist, AbstractHypersphericalDistribution) and dist.dim == 2
        ), "dist must be a distribution on the sphere."
        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_cart(
            dist.pdf, degree, transformation
        )
        return shd

    @staticmethod
    def _fun_cart_to_fun_sph(fun_cart):
        """Convert a function using Cartesian coordinates to one using spherical coordinates."""

        def fun_sph(phi, theta):
            x, y, z = AbstractSphericalDistribution.sph_to_cart(
                np.ravel(phi), np.ravel(theta)
            )
            vals = fun_cart(column_stack((x, y, z)))
            return reshape(vals, shape(theta))

        return fun_sph

    @staticmethod
    def from_function_via_integral_cart(fun_cart, degree, transformation="identity"):
        fun_sph = SphericalHarmonicsDistributionComplex._fun_cart_to_fun_sph(fun_cart)
        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_sph(
            fun_sph, degree, transformation
        )
        return shd

    @staticmethod
    def from_function_via_integral_sph(fun, degree, transformation="identity"):
        if transformation == "sqrt":
            raise NotImplementedError("Transformations are not supported yet")
        if transformation == "identity":
            fun_with_trans = fun
        else:
            raise ValueError("Transformation not supported")

        coeff_mat = full((degree + 1, 2 * degree + 1), float('NaN'), dtype=complex)

        def real_part(phi, theta, n, m):
            return real(
                fun_with_trans(array(phi), array(theta))
                * conj(sph_harm(m, n, phi, theta))
                * sin(theta)
            )

        def imag_part(phi, theta, n, m):
            return imag(
                fun_with_trans(array(phi), array(theta))
                * conj(sph_harm(m, n, phi, theta))
                * sin(theta)
            )

        for n in range(degree + 1):  # Use n instead of l to comply with PEP 8
            for m in range(-n, n + 1):
                real_integral, _ = scipy.integrate.nquad(
                    real_part, [[0, 2 * pi], [0, pi]], args=(n, m)
                )
                imag_integral, _ = scipy.integrate.nquad(
                    imag_part, [[0, 2 * pi], [0, pi]], args=(n, m)
                )

                if isnan(real_integral) or isnan(imag_integral):
                    print(f"Integration failed for l={n}, m={m}")

                coeff_mat[n, m + n] = real_integral + 1j * imag_integral

        return SphericalHarmonicsDistributionComplex(coeff_mat, transformation)