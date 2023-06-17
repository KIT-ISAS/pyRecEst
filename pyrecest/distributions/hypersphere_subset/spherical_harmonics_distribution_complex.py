from .abstract_spherical_harmonics_distribution import AbstractSphericalHarmonicsDistribution
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
import numpy as np
# pylint: disable=E0611
from scipy.special import sph_harm
import scipy
from .spherical_grid_distribution import SphericalGridDistribution
import pyshtools as pysh
from .abstract_spherical_distribution import AbstractSphericalDistribution
from .custom_hyperspherical_distribution import CustomHypersphericalDistribution

class SphericalHarmonicsDistributionComplex(AbstractSphericalHarmonicsDistribution):
    def __init__(self, coeff_mat, transformation='identity'):
        AbstractSphericalHarmonicsDistribution.__init__(self, coeff_mat, transformation)
    
    def value(self, xs):
        xs = np.atleast_2d(xs)
        vals = np.zeros(xs.shape[0], dtype=np.complex128)
        phi, theta = AbstractSphereSubsetDistribution.cart_to_sph(xs[:,0], xs[:,1], xs[:,2])#, mode="elevation")
        
          
        for l_curr in range(self.coeff_mat.shape[0]):
            for m_curr in range(-l_curr, l_curr + 1):
                # Evaluate it for all query points at once
                y_lm = sph_harm(m_curr, l_curr, phi, theta)
                vals += self.coeff_mat[l_curr, l_curr + m_curr] * y_lm
        """
        for l_curr in range(self.coeff_mat.shape[0]):
            for m_curr in range(-l_curr, l_curr + 1):
                # Evaluate it for all query points at once
                y_lm = sph_harm(m_curr, l_curr, theta, phi)  # swapping phi and theta
                vals += self.coeff_mat[l_curr, l_curr + m_curr] * y_lm
        """  
        assert np.all(np.abs(np.imag(vals))<1e-10), "Coefficients apparently do not represent a real function."
        return np.real(vals)
    
    

    def to_spherical_harmonics_distribution_real(self):
        from .spherical_harmonics_distribution_real import SphericalHarmonicsDistributionReal
        
        if self.transformation != 'identity':
            raise ValueError('Transformation currently not supported')
        
        coeff_mat_real = np.empty(self.coeff_mat.shape, dtype=np.float64)
        
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
    def from_grid(gridValues, grid=None, transformation='identity', degree=None):
        # Values are assumed to be given without any transformation.
        # If no grid is given (i.e., only one input or an empty grid),
        # a grid that is directly compatible with
        # the spherical harmonics transform is assumed. For other
        # regular grids, directly provide grid (it is treated as
        # irregular grid then).

        if transformation == 'sqrt':
            gridValues = np.sqrt(gridValues)
        
        if grid is None or len(grid) == 0:
            assert len(gridValues[0]) > 1, 'For regular grids, provide values as a matrix.'
            if degree is None:
                degree = (-6 + np.sqrt(36 - 8 * (4 - len(gridValues)))) / 4
            assert degree == np.floor(degree), \
                'Based on the number of values, this grid is definitely not directly compatible with spherical harmonics transform.'
            grid_values_reshaped = np.reshape(gridValues, (degree+2, 2*degree+2))
            clm = pysh.SHGrid.from_array(grid_values_reshaped).expand()
        else:
            assert len(gridValues) == np.prod(gridValues.shape) and len(gridValues) == len(grid[0]), \
                'For irregular grids, provide values and grid points as 1D arrays.'
            lon, lat = np.arctan2(np.sqrt(grid[0]**2 + grid[1]**2), grid[2]), np.arctan2(grid[1], grid[0])
            lon, lat = np.degrees(lon), np.degrees(lat)
            grid = pysh.SHGrid.from_array(gridValues)
            clm = grid.expand()

        return SphericalHarmonicsDistributionComplex(clm, transformation=transformation)
    
    @staticmethod
    def from_distribution(dist, degree, transformation='identity'):
        if isinstance(dist, SphericalGridDistribution):
            shd = SphericalHarmonicsDistributionComplex.from_grid(dist.fvals, dist.grid, transformation)
        else:
            shd = SphericalHarmonicsDistributionComplex.from_distribution_fast(dist, degree, transformation)
        return shd

    @staticmethod
    def from_distribution_fast(dist, degree, transformation='identity'):
        assert(isinstance(dist, AbstractHypersphericalDistribution) and dist.dim == 3), 'dist must be a distribution on the sphere.'
        shd = SphericalHarmonicsDistributionComplex.from_function_fast(dist.pdf, degree, transformation)
        return shd
    
    @staticmethod
    def from_function(fun, degree, transformation='identity'):
        return SphericalHarmonicsDistributionComplex.from_function_fast(fun, degree, transformation)
    
    @staticmethod
    def from_function_fast(fun, degree, transformation='identity'):
        # Default to Cartesian variant
        SphericalHarmonicsDistributionComplex.from_function_fast_cart(fun, degree, transformation)

    @staticmethod
    def _fun_cart_to_fun_sph(fun_cart):
        """ Convert a function using Cartesian coordinates to one using spherical coordinates."""
        def fun_sph(phi, theta):
            x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())
            vals = fun_cart(np.column_stack((x,y,z)))
            return np.reshape(vals, np.shape(theta))        
        
        return fun_sph

    @staticmethod
    def from_function_fast_cart(fun_cart, degree, transformation='identity'):
        assert degree >= 1
        
        fun_sph = SphericalHarmonicsDistributionComplex._fun_cart_to_fun_sph(fun_cart)
        return SphericalHarmonicsDistributionComplex.from_function_fast_sph(fun_sph, degree, transformation)
        
    @staticmethod
    def from_function_fast_sph(fun_sph, degree, transformation='identity'):
        assert(degree >= 1)
        lat = np.linspace(0, 2*np.pi, 2*degree+2)
        lon = np.linspace(np.pi/2, -np.pi/2, degree+2)
        latMesh, lonMesh = np.meshgrid(lat, lon)
        fval = fun_sph(latMesh, lonMesh)
        if transformation == 'sqrt':
            fval = np.sqrt(fval)
        elif transformation != 'identity':
            raise ValueError('Currently, only identity transformation is supported')
        ### think twice if 
        plm = pysh.expand.SHExpandDH(fval)
        ########### leopardi
        n_lat = 2*degree + 2
        n_lon = n_lat//2
        lat = np.linspace(90, -90, n_lat)
        lon = np.linspace(0, 360, n_lon, endpoint=False)
        latMesh, lonMesh = np.meshgrid(lat, lon)
        fval = fun_sph(latMesh, lonMesh)
        if transformation == 'sqrt':
            fval = np.sqrt(fval)
        elif transformation != 'identity':
            raise ValueError('Currently, only identity transformation is supported')
        cilm = pysh.expand.SHExpandDH(fval, sampling=2)
        complex_cilm = cilm[0,:,:] + 1j * cilm[1,:,:]
        1+1 #TODO
        ##
        ###
        return shd
    

    @staticmethod
    def from_function_via_integral_cart(fun_cart, degree, transformation='identity'):
        fun_sph = SphericalHarmonicsDistributionComplex._fun_cart_to_fun_sph(fun_cart)
        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_sph(fun_sph, degree, transformation)
        
        ## testing
        #for l in range(degree + 1):
        #    for m in range(-l, l + 1):
        #        harmonics_cart = SphericalHarmonicsDistributionComplex._fun_cart_to_fun_sph(fun_cart)
        #        lambda x: fun_cart(x) * np.conj(sph_harm(m, l, x[0], x[1]))
        
        ###
        
        
        return shd
    
    @staticmethod
    def from_function_via_integral_sph(fun, degree, transformation='identity'):
        if transformation == 'sqrt':
            fun_with_trans = lambda theta, phi: np.sqrt(fun(theta, phi))
        elif transformation == 'identity':
            fun_with_trans = fun
        else:
            raise ValueError('Transformation not supported')

        coeff_mat = np.full((degree+1, 2*degree+1), np.nan, dtype=complex)

        def real_part(phi, theta, l, m):
            #return np.real(fun_with_trans(np.array(theta), np.array(phi)) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(phi))
            return np.real(fun_with_trans(np.array(theta), np.array(phi)) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(theta))
            #return np.real(fun_with_trans(np.array(theta), np.array(phi)) * np.conj(sph_harm(m, l, theta, phi)) * np.sin(phi))
            #return np.real(fun_with_trans(np.array(theta), np.array(phi)) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(theta))
            #return np.real(fun_with_trans(np.array(phi), np.array(theta)) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(theta))

        def imag_part(phi, theta, l, m):
            #return np.imag(fun_with_trans(np.array(theta), np.array(phi)) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(phi))
            return np.imag(fun_with_trans(np.array(theta), np.array(phi)) * np.conj(sph_harm(m, l, phi, theta)) * np.sin(theta))
            #return np.imag(fun_with_trans(np.array(theta), np.array(phi)) * np.conj(sph_harm(m, l, theta, phi)) * np.sin(phi))
            #return np.imag(fun_with_trans(np.array(theta), np.array(phi)) * np.conj(sph_harm(m, l, theta, phi)) * np.sin(theta))
            #return np.imag(fun_with_trans(np.array(phi), np.array(theta)) * np.conj(sph_harm(m, l, theta, phi)) * np.sin(theta))

        for l in range(degree + 1):
            for m in range(-l, l + 1):
                real_integral, _ = scipy.integrate.nquad(real_part, [[0, np.pi], [0, 2*np.pi]], args=(l, m))
                imag_integral, _ = scipy.integrate.nquad(imag_part, [[0, np.pi], [0, 2*np.pi]], args=(l, m))
                
                if np.isnan(real_integral) or np.isnan(imag_integral):
                    print(f"Integration failed for l={l}, m={m}")
                    
                coeff_mat[l, m+l] = real_integral + 1j*imag_integral

        return SphericalHarmonicsDistributionComplex(coeff_mat, transformation)


