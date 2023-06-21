import numpy as np
import unittest
import warnings
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_real import SphericalHarmonicsDistributionReal
from pyrecest.distributions.hypersphere_subset.abstract_spherical_distribution import AbstractSphericalDistribution
from parameterized import parameterized


class SphericalHarmonicsDistributionRealTest(unittest.TestCase):
    def testNormalizationError(self):
        self.assertRaises(ValueError, SphericalHarmonicsDistributionReal, 0)

    def testNormalizationWarning(self):
        with warnings.catch_warnings(record=True) as w:
            SphericalHarmonicsDistributionReal(np.random.rand(3, 5))
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
        
    def testNormalization(self):
        unnormalized_coeffs = np.random.rand(3, 5)
        shd = SphericalHarmonicsDistributionReal(unnormalized_coeffs)
        self.assertAlmostEqual(shd.integrate(), 1, delta=1e-6)

        phi, theta = np.random.rand(10) * 2 * np.pi, np.random.rand(10) * np.pi - np.pi / 2
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi, theta)
        vals_normalized = shd.pdf(np.column_stack((x,y,z)))
        shd.coeff_mat = unnormalized_coeffs
        vals_unnormalized = shd.pdf(np.column_stack((x,y,z)))
        self.assertTrue(np.allclose(np.diff(vals_normalized / vals_unnormalized), np.zeros((1, x.size - 1)), atol=1e-6))
 
    @parameterized.expand([
        ("l0m0", [[1, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 0]], lambda x, _, __: np.ones_like(x)*np.sqrt(1/(4 * np.pi))),
        ("l1mneg1", [[0, np.nan, np.nan, np.nan, np.nan], [1, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 0]], lambda _, y, __: np.sqrt(3/(4 * np.pi)) * y),
        ("l1_m0", [[0, np.nan, np.nan, np.nan, np.nan], [0, 1, 0, np.nan, np.nan], [0, 0, 0, 0, 0]], lambda _,__,z: np.sqrt(3/(4 * np.pi)) * z),
        ("l1_m1", [[0, np.nan, np.nan, np.nan, np.nan], [0, 0, 1, np.nan, np.nan], [0, 0, 0, 0, 0]], lambda x,_,__: np.sqrt(3/(4 * np.pi)) * x),
        ("l2_mneg2", [[0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [1, 0, 0, 0, 0]], lambda x, y,__: 1/2 * np.sqrt(15/np.pi) * x * y),
        ("l2_mneg1", [[0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 1, 0, 0, 0]], lambda _,y, z: 1/2 * np.sqrt(15/np.pi) * y * z),
        ("l2_m0", [[0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 0, 1, 0, 0]], lambda x, y, z: 1/4 * np.sqrt(5/np.pi) * (2 * z**2 - x**2 - y**2)),
        ("l2_m1", [[0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 0, 0, 1, 0]], lambda x,_, z: 1/2 * np.sqrt(15/np.pi) * x * z),
        ("l2_m2", [[0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 1]], lambda x, y,_: 1/4 * np.sqrt(15/np.pi) * (x**2 - y**2))
    ])
    def test_basis_function(self, name, coeff_mat, result_func):
        np.random.seed(10)
        shd = SphericalHarmonicsDistributionReal(1/np.sqrt(4*np.pi))
        shd.coeff_mat = np.array(coeff_mat)
        phi, theta = np.random.rand(10) * 2 * np.pi, np.random.rand(10) * np.pi - np.pi/2
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi, theta)
        np.testing.assert_allclose(shd.pdf(np.column_stack((x,y,z))), result_func(x,y,z), rtol=1e-6, err_msg=name)

    @parameterized.expand([
        ("l0_m0", np.array([[1, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 0]])),
        ("l1_mneg1", np.array([[1, np.nan, np.nan, np.nan, np.nan], [1, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 0]])),
        ("l1_m0", np.array([[1, np.nan, np.nan, np.nan, np.nan], [0, 1, 0, np.nan, np.nan], [0, 0, 0, 0, 0]])),
        ("l1_m1", np.array([[1, np.nan, np.nan, np.nan, np.nan], [0, 0, 1, np.nan, np.nan], [0, 0, 0, 0, 0]])),
        ("l2_mneg2", np.array([[1, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [1, 0, 0, 0, 0]])),
        ("l2_mneg1", np.array([[1, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 1, 0, 0, 0]])),
        ("l2_m0", np.array([[1, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 0, 1, 0, 0]])),
        ("l2_m1", np.array([[1, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 0, 0, 1, 0]])),
        ("l2_m2", np.array([[1, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 1]])),
        ("l3_mneg3", np.array([[1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0, 0, np.nan, np.nan], [1, 0, 0, 0, 0, 0, 0]])),
        ("l3_mneg2", np.array([[1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0, 0, np.nan, np.nan], [0, 1, 0, 0, 0, 0, 0]])),
        ("l3_mneg1", np.array([[1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0, 0, np.nan, np.nan], [0, 0, 1, 0, 0, 0, 0]])),
        ("l3_m0", np.array([[1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0, 0, np.nan, np.nan], [0, 0, 0, 1, 0, 0, 0]])),
        ("l3_m1", np.array([[1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 1, 0, 0]])),
        ("l3_m2", np.array([[1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 0, 1, 0]])),
        ("l3_m3", np.array([[1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, np.nan, np.nan, np.nan, np.nan], [0, 0, 0, 0, 0, np.nan, np.nan], [0, 0, 0, 0, 0, 0, 1]])),
        ("random", np.random.rand(4, 7))
    ])
    def test_conversion(self, _, coeff_mat):
        rshd = SphericalHarmonicsDistributionReal(coeff_mat)
        cshd = rshd.to_spherical_harmonics_distribution_complex()
        phi_to_test, theta_to_test = np.random.rand(10) * 2 * np.pi, np.random.rand(10) * np.pi - np.pi / 2
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi_to_test, theta_to_test)
        np.testing.assert_allclose(cshd.pdf(np.column_stack((x, y, z))), rshd.pdf(np.column_stack((x, y, z))), atol=1e-6)

    def test_conversion_to_complex_and_back(self):
        # Suppress warnings related to normalization
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rshd = SphericalHarmonicsDistributionReal(np.random.rand(4, 7))

        cshd = rshd.to_spherical_harmonics_distribution_complex()
        rshd2 = cshd.to_spherical_harmonics_distribution_real()
        np.testing.assert_allclose(rshd2.coeff_mat, rshd.coeff_mat, atol=1e-6, equal_nan=True)

    def test_integral_analytical(self):
        # Suppress warnings related to normalization
        unnormalized_coeffs = np.random.rand(3, 5)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shd = SphericalHarmonicsDistributionReal(unnormalized_coeffs)

        np.testing.assert_allclose(shd.integrate_numerically(), shd.integrate(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()


