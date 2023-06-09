import numpy as np

# pylint: disable=E0611
from scipy.special import sph_harm

from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution
from .abstract_spherical_harmonics_distribution import (
    AbstractSphericalHarmonicsDistribution,
)


class SphericalHarmonicsDistributionReal(AbstractSphericalHarmonicsDistribution):
    def __init__(self, coeff_mat, transformation="identity"):
        if not np.all(np.isreal(coeff_mat)):
            raise ValueError("Coefficients must be real")
        AbstractSphericalHarmonicsDistribution.__init__(self, coeff_mat, transformation)

    @staticmethod
    def real_spherical_harmonic_basis_function(n, m, theta, phi):
        y_lm = sph_harm(m, n, phi, theta)

        if m < 0:
            y_nm_real = -np.sqrt(2) * np.imag(y_lm)
        elif m == 0:
            y_nm_real = np.real(y_lm)
        else:
            y_nm_real = (-1) ** m * np.sqrt(2) * np.real(y_lm)

        return y_nm_real

    def value(self, xs):
        xs = np.atleast_2d(xs)
        vals = np.zeros(xs.shape[0])
        phi, theta = AbstractSphereSubsetDistribution.cart_to_sph(
            xs[:, 0], xs[:, 1], xs[:, 2]
        )

        for n_curr in range(self.coeff_mat.shape[0]):
            for m_curr in range(-n_curr, n_curr + 1):
                # Evaluate it for all query points at once
                y_lm_real = SphericalHarmonicsDistributionReal.real_spherical_harmonic_basis_function(
                    n_curr, m_curr, theta, phi
                )
                vals += self.coeff_mat[n_curr, n_curr + m_curr] * y_lm_real

        return vals

    def to_spherical_harmonics_distribution_complex(self):
        from .spherical_harmonics_distribution_complex import (
            SphericalHarmonicsDistributionComplex,
        )

        if self.transformation != "identity":
            raise NotImplementedError("Transformation currently not supported")

        real_coeff_mat = self.coeff_mat
        complex_coeff_mat = np.full_like(real_coeff_mat, np.nan, dtype=complex)

        for n in range(real_coeff_mat.shape[0]):
            for m in range(-n, n + 1):
                if m < 0:
                    complex_coeff_mat[n, n + m] = (
                        1j * real_coeff_mat[n, n + m] + real_coeff_mat[n, n - m]
                    ) / np.sqrt(2)
                elif m > 0:
                    complex_coeff_mat[n, n + m] = (
                        (-1) ** m
                        * (-1j * real_coeff_mat[n, n - m] + real_coeff_mat[n, n + m])
                        / np.sqrt(2)
                    )
                else:  # m == 0
                    complex_coeff_mat[n, n] = real_coeff_mat[n, n]

        return SphericalHarmonicsDistributionComplex(
            complex_coeff_mat, self.transformation
        )
