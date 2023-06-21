""" Do not check in, not working yet!
import unittest
import numpy as np
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_real import SphericalHarmonicsDistributionReal
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import SphericalHarmonicsDistributionComplex

class AbstractSphericalHarmonicsDistributionTest(unittest.TestCase):

    def test_convolution_dim_and_mean_and_normalization(self):
        coeffMat = np.random.rand(7, 13)
        coeffMat[0, 0] = 1 / np.sqrt(4 * np.pi)
        rshdRandom = SphericalHarmonicsDistributionReal(coeffMat)
        coeffMatZonal = np.zeros((7, 13))
        coeffMatZonal[0, 0] = 1 / np.sqrt(4 * np.pi)
        coeffMatZonal[1:, 1:2:] = np.random.rand(6, 1) # Needs this shape to work properly
        cshdZonal = SphericalHarmonicsDistributionComplex(coeffMatZonal)

        # Test both for real and complex
        shdRandoms = [rshdRandom, rshdRandom.to_spherical_harmonics_distribution_complex()]
        shdZonals = [cshdZonal.to_spherical_harmonics_distribution_real(), cshdZonal]
        for i in range(2):
            shdRandomCurr = shdRandoms[i]
            shdZonalCurr = shdZonals[i]
            shdtmp = shdRandomCurr.convolve(shdZonalCurr)
            self.assertEqual(shdtmp.coeff_mat.shape, (7, 13))
            np.testing.assert_allclose(shdtmp.mean_direction(), shdRandomCurr.mean_direction(), atol=1e-10)

            shdtmp = shdRandomCurr.convolve(shdZonalCurr.truncate(5))
            self.assertEqual(shdtmp.coeff_mat.shape, (6, 11))
            np.testing.assert_allclose(shdtmp.mean_direction(), shdRandomCurr.mean_direction(), atol=1e-10)
            np.testing.assert_allclose(shdtmp.integralNumerical(), 1, atol=1e-4)

            shdtmp = shdRandomCurr.convolve(shdZonalCurr.truncate(4))
            self.assertEqual(shdtmp.coeff_mat.shape, (5, 9))
            np.testing.assert_allclose(shdtmp.mean_direction(), shdRandomCurr.mean_direction(), atol=1e-10)
            np.testing.assert_allclose(shdtmp.integralNumerical(), 1, atol=1e-4)

            shdRandomTrunc = shdRandomCurr.truncate(5)
            shdtmp = shdRandomTrunc.convolve(shdZonalCurr)
            self.assertEqual(shdtmp.coeff_mat.shape, (6, 11))
            np.testing.assert_allclose(shdtmp.mean_direction(), shdRandomCurr.mean_direction(), atol=1e-10)
            np.testing.assert_allclose(shdtmp.integralNumerical(), 1, atol=1e-4)

            shdRandomTrunc = shdRandomCurr.truncate(4)
            shdtmp = shdRandomTrunc.convolve(shdZonalCurr)
            self.assertEqual(shdtmp.coeff_mat.shape, (5, 9))
            np.testing.assert_allclose(shdtmp.mean_direction(), shdRandomCurr.mean_direction(), atol=1e-10)
            np.testing.assert_allclose(shdtmp.integralNumerical(), 1, atol=1e-4)

    

if __name__ == '__main__':
    unittest.main()
"""