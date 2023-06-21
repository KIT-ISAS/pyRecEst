from .abstract_spherical_harmonics_distribution import AbstractSphericalHarmonicsDistribution
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
import numpy as np
# pylint: disable=E0611
from scipy.special import sph_harm
import scipy
from .abstract_spherical_distribution import AbstractSphericalDistribution


class SphericalHarmonicsDistributionComplex(AbstractSphericalHarmonicsDistribution):
    def __init__(self, coeff_mat, transformation='identity', assert_real = True):
        AbstractSphericalHarmonicsDistribution.__init__(self, coeff_mat, transformation)
        self.assert_real = assert_real
    
    def value(self, xs):
        xs = np.atleast_2d(xs)
        phi, theta = AbstractSphereSubsetDistribution.cart_to_sph(xs[:,0], xs[:,1], xs[:,2])
        return self.value_sph(phi, theta)
        
    def value_sph(self, phi, theta):
        vals = np.zeros(theta.shape[0], dtype=np.complex128)
        for l_curr in range(self.coeff_mat.shape[0]):
            for m_curr in range(-l_curr, l_curr + 1):
                # Evaluate it for all query points at once
                # scipy's sph_harm uses the opposite convention for phi and theta!
                # This is correct since sph_harm expects theta (our phi) and then phi (our theta) 
                y_lm = sph_harm(m_curr, l_curr, phi, theta)
                vals += self.coeff_mat[l_curr, l_curr + m_curr] * y_lm
        
        if self.assert_real:
            assert np.all(np.abs(np.imag(vals))<1e-10), "Coefficients apparently do not represent a real function."
            return np.real(vals)
        
        return vals
    
    def to_spherical_harmonics_distribution_real(self):
        from .spherical_harmonics_distribution_real import SphericalHarmonicsDistributionReal
        
        if self.transformation != 'identity':
            raise ValueError('Transformation currently not supported')
        
        coeff_mat_real = np.empty(self.coeff_mat.shape, dtype=np.float)
        
        coeff_mat_real[0, 0] = self.coeff_mat[0, 0]

        for l in range(1, self.coeff_mat.shape[0]):
            for m in range(-l, l+1):
                if m < 0:
                    coeff_mat_real[l, l+m] = (-1)**m * np.sqrt(2) * (-1 if (-m)%2 else 1) * np.imag(self.coeff_mat[l, l+m])
                elif m > 0:
                    coeff_mat_real[l, l+m] = np.sqrt(2) * (-1 if m%2 else 1) * np.real(self.coeff_mat[l, l+m])
                else:  # m == 0
                    coeff_mat_real[l, l] = self.coeff_mat[l, l]


        shd = SphericalHarmonicsDistributionReal(np.real(coeff_mat_real), self.transformation)
        
        return shd
    
    def mean_direction(self):
        if np.prod(self.coeff_mat.shape) <= 1:
            raise ValueError('Too few coefficients available to calculate the mean')

        y = np.imag(self.coeff_mat[1, 0] + self.coeff_mat[1, 2]) / np.sqrt(2)
        x = np.real(self.coeff_mat[1, 0] - self.coeff_mat[1, 2]) / np.sqrt(2)
        z = np.real(self.coeff_mat[1, 1])

        if np.linalg.norm(np.array([x, y, z])) < 1e-9:
            raise ValueError('Coefficients of degree 1 are almost zero. Therefore, no meaningful mean is available')

        mu = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))

        return mu
    
    @staticmethod
    def from_distribution_via_integral(dist, degree, transformation='identity'):
        assert(isinstance(dist, AbstractHypersphericalDistribution) and dist.dim == 2), 'dist must be a distribution on the sphere.'
        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_cart(dist.pdf, degree, transformation)
        return shd
    
    @staticmethod
    def _fun_cart_to_fun_sph(fun_cart):
        """ Convert a function using Cartesian coordinates to one using spherical coordinates."""
        def fun_sph(phi, theta):
            x, y, z = AbstractSphericalDistribution.sph_to_cart(np.ravel(phi), np.ravel(theta))
            vals = fun_cart(np.column_stack((x,y,z)))
            return np.reshape(vals, np.shape(theta))        
        
        return fun_sph

    @staticmethod
    def from_function_via_integral_cart(fun_cart, degree, transformation='identity'):
        fun_sph = SphericalHarmonicsDistributionComplex._fun_cart_to_fun_sph(fun_cart)
        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_sph(fun_sph, degree, transformation)
        return shd
    
    @staticmethod
    def from_function_via_integral_sph(fun, degree, transformation='identity'):
        if transformation == 'sqrt':
            raise NotImplementedError('Transformations are not supported yet')
        elif transformation == 'identity':
            fun_with_trans = fun
        else:
            raise ValueError('Transformation not supported')

        coeff_mat = np.full((degree+1, 2*degree+1), np.nan, dtype=complex)

        def real_part(phi, theta, l, m):
            return np.real(fun_with_trans(np.array(phi), np.array(theta)) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(theta))

        def imag_part(phi, theta, l, m):
            return np.imag(fun_with_trans(np.array(phi), np.array(theta)) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(theta))

        for l in range(degree + 1):
            for m in range(-l, l + 1):
                real_integral, _ = scipy.integrate.nquad(real_part, [[0, 2*np.pi], [0, np.pi]], args=(l, m))
                imag_integral, _ = scipy.integrate.nquad(imag_part, [[0, 2*np.pi], [0, np.pi]], args=(l, m))
                
                if np.isnan(real_integral) or np.isnan(imag_integral):
                    print(f"Integration failed for l={l}, m={m}")
                    
                coeff_mat[l, m+l] = real_integral + 1j*imag_integral

        return SphericalHarmonicsDistributionComplex(coeff_mat, transformation)