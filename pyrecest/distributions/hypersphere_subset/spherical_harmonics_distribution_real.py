from .abstract_spherical_harmonics_distribution import AbstractSphericalHarmonicsDistribution
import numpy as np
# pylint: disable=E0611
from scipy.special import sph_harm
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution

class SphericalHarmonicsDistributionReal(AbstractSphericalHarmonicsDistribution):
    def __init__(self, coeff_mat, transformation='identity'):
        if not np.all(np.isreal(coeff_mat)):
            raise ValueError("Coefficients must be real")
        AbstractSphericalHarmonicsDistribution.__init__(self, coeff_mat, transformation)

    @staticmethod
    def real_spherical_harmonic(l, m, theta, phi):
        y_lm = sph_harm(m, l, phi, theta)
        
        if m < 0:
            y_lm_real = -np.sqrt(2) * np.imag(y_lm)
        elif m == 0:
            y_lm_real = np.real(y_lm)
        else:
            y_lm_real = (-1) ** m * np.sqrt(2) * np.real(y_lm)
        
        return y_lm_real
    
    def value(self, xs):
        xs = np.atleast_2d(xs)
        vals = np.zeros(xs.shape[0])
        phi, theta = AbstractSphereSubsetDistribution.cart_to_sph(xs[:,0], xs[:,1], xs[:,2])
        
        for l_curr in range(self.coeff_mat.shape[0]):
            for m_curr in range(-l_curr, l_curr + 1):
                # Evaluate it for all query points at once
                y_lm_real = SphericalHarmonicsDistributionReal.real_spherical_harmonic(l_curr, m_curr, theta, phi)
                vals += self.coeff_mat[l_curr, l_curr + m_curr] * y_lm_real
            
        return vals
    
    def to_spherical_harmonics_distribution_complex(self):
        from .spherical_harmonics_distribution_complex import SphericalHarmonicsDistributionComplex
        if self.transformation != 'identity':
            raise NotImplementedError("Transformation currently not supported")

        real_coeff_mat = self.coeff_mat
        complex_coeff_mat = np.full_like(real_coeff_mat, np.nan, dtype=np.complex128)
        
        for l in range(real_coeff_mat.shape[0]):
            for m in range(-l, l+1):
                if m < 0:
                    complex_coeff_mat[l, l+m] = (1j * real_coeff_mat[l, l+m] + real_coeff_mat[l, l-m]) / np.sqrt(2)
                elif m > 0:
                    complex_coeff_mat[l, l+m] = (-1)**m * (-1j * real_coeff_mat[l, l-m] + real_coeff_mat[l, l+m]) / np.sqrt(2)
                else:  # m == 0
                    complex_coeff_mat[l, l] = real_coeff_mat[l, l]
                    
        return SphericalHarmonicsDistributionComplex(complex_coeff_mat, self.transformation)
