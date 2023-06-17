import numpy as np
from .abstract_spherical_distribution import AbstractSphericalDistribution
from scipy.linalg import norm
import copy
import warnings
from ..abstract_orthogonal_basis_distribution import AbstractOrthogonalBasisDistribution

class AbstractSphericalHarmonicsDistribution(AbstractSphericalDistribution, AbstractOrthogonalBasisDistribution):
    def __init__(self, coeff_mat, transformation='identity'):
        AbstractSphericalDistribution.__init__(self)
        coeff_mat = np.atleast_2d(coeff_mat)
        assert coeff_mat.shape[1] == coeff_mat.shape[0] * 2 - 1, "CoefficientMatrix:Size, Dimensions of coefficient Matrix are incompatible."
        
        # Ignore irrelevant entries of coeff_mat and set to NaN
        n = coeff_mat.shape[0]
        coeff_mat = coeff_mat + np.block([
            [np.zeros((n-1, 1)), np.kron(np.triu(np.full((n-1, n-1), np.nan)), np.array([[1, 1]]))],
            [np.zeros((1, 2*n-1))]
        ])
        AbstractOrthogonalBasisDistribution.__init__(self, coeff_mat, transformation) 
    
    def pdf(self, xs):
        return AbstractOrthogonalBasisDistribution.pdf(self, xs)
    
    def normalize_in_place(self):
        int_val = self.integrate()
        if int_val < 0:
            warnings.warn("Warning: Normalization:negative - Coefficient for first degree is negative. This can either be caused by a user error or due to negativity caused by non-square rooted version")
        elif np.abs(int_val) < 1e-12:
            raise ValueError("Normalization:almostZero - Coefficient for first degree is too close to zero, this usually points to a user error")
        elif np.abs(int_val - 1) > 1e-5:
            warnings.warn("Warning: Normalization:notNormalized - Coefficients apparently do not belong to normalized density. Normalizing...")
        else:
            return

        if self.transformation == 'identity':
            self.coeff_mat = self.coeff_mat / int_val
        elif self.transformation == 'sqrt':
            self.coeff_mat = self.coeff_mat / np.sqrt(int_val)
        else:
            warnings.warn("Warning: Currently cannot normalize")
            
        return self
    
    def convolve(self, other):
        """Convolves this distribution with another one."""
        
        # Check if other is zonal. For this, only the coefficients for m=0 are allowed to be nonzero.
        zonal_test_mat = copy.copy(other.coeff_mat)
        np.fill_diagonal(zonal_test_mat, np.nan)
        assert np.all((np.abs(zonal_test_mat) <= 1E-5) | np.isnan(zonal_test_mat)), 'Other is not zonal.'

        current = copy.deepcopy(self)
        # Truncate to the smaller degree
        if other.coeff_mat.shape[0] < self.coeff_mat.shape[0]:
            current = current.truncate(other.coeff_mat.shape[0]-1)
        elif self.coeff_mat.shape[0] < other.coeff_mat.shape[0]:
            other = other.truncate(self.coeff_mat.shape[0]-1)

        if self.transformation == 'identity':
            # Get coefficients of other of order 0. Use broadcasting.
            new_coeff_mat = self.coeff_mat \
                * np.expand_dims(other.coeff_mat[np.arange(other.coeff_mat.shape[0]), np.arange(other.coeff_mat.shape[0])] \
                * np.sqrt(4*np.pi/(2*np.arange(other.coeff_mat.shape[0])+1)), axis=1)
            
            shd = self.__class__(new_coeff_mat, self.transformation)  # Do not use constructor to allow for inheritability
        else:
            raise ValueError('Transformation not supported')

        return shd

    def integrate(self):
            
        if self.transformation == 'identity':
            int_val = self.coeff_mat[0, 0] * np.sqrt(4 * np.pi)
        elif self.transformation == 'sqrt':
            int_val = norm(self.coeff_mat[~np.isnan(self.coeff_mat)])**2
        else:
            raise ValueError("No analytical formula for normalization available")
        
        assert np.abs(np.imag(int_val)<1e-8)
        return np.real(int_val)
        
    def truncate(self, degree):
        result = copy.deepcopy(self)
        if result.coeff_mat.shape[0] - 1 > degree:
            result.coeff_mat = result.coeff_mat[:degree+1, :2*degree+1]
        elif result.coeff_mat.shape[0] - 1 < degree:
            warnings.warn("Less coefficients than desired, filling up with zeros")
            new_coeff_mat = np.zeros((degree+1, 2*degree+1), dtype=self.coeff_mat.dtype)
            new_coeff_mat[:result.coeff_mat.shape[0], :2*result.coeff_mat.shape[0]-1] = result.coeff_mat
            for i in range(new_coeff_mat.shape[0] - 1):
                new_coeff_mat[i, 2*i+1:] = np.nan
            result.coeff_mat = new_coeff_mat
        
        return result
