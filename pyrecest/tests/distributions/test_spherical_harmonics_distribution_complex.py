import unittest

import numpy as np
from parameterized import parameterized
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.abstract_spherical_distribution import (
    AbstractSphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)


class SphericalHarmonicsDistributionComplexTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        coeff_rand = np.random.rand(9)
        self.unnormalized_coeffs = np.array(
            [
                [coeff_rand[0], np.nan, np.nan, np.nan, np.nan],
                [
                    coeff_rand[1] + 1j * coeff_rand[2],
                    coeff_rand[3],
                    -coeff_rand[1] + 1j * coeff_rand[2],
                    np.nan,
                    np.nan,
                ],
                [
                    coeff_rand[4] + 1j * coeff_rand[5],
                    coeff_rand[6] + 1j * coeff_rand[7],
                    coeff_rand[8],
                    -coeff_rand[6] + 1j * coeff_rand[7],
                    coeff_rand[4] - 1j * coeff_rand[5],
                ],
            ]
        )

    def test_mormalization_error(self):
        self.assertRaises(ValueError, SphericalHarmonicsDistributionComplex, 0)

    def test_normalization(self):
        with self.assertWarns(Warning):
            shd = SphericalHarmonicsDistributionComplex(self.unnormalized_coeffs)

        self.assertAlmostEqual(shd.integrate(), 1, delta=1e-5)

        # Enforce unnormalized coefficients and compare ratio
        phi, theta = (
            np.random.rand(1, 10) * 2 * np.pi,
            np.random.rand(1, 10) * np.pi - np.pi / 2,
        )
        x, y, z = np.array(
            [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
        )
        vals_normalized = shd.pdf(np.column_stack([x, y, z]))
        shd.coeff_mat = self.unnormalized_coeffs
        vals_unnormalized = shd.pdf(np.column_stack([x, y, z]))
        self.assertTrue(
            np.allclose(
                np.diff(vals_normalized / vals_unnormalized),
                np.zeros(vals_normalized.shape[0] - 1),
                atol=1e-6,
            )
        )

    @parameterized.expand([("identity",), ("sqrt",)])
    def test_integral_analytical(self, transformation):
        """Test if the analytical integral is equal to the numerical integral"""
        np.random.seed(10)
        coeff_rand = np.random.rand(1, 9)
        unnormalized_coeffs = np.array(
            [
                [coeff_rand[0, 0], np.nan, np.nan, np.nan, np.nan],
                [
                    coeff_rand[0, 1] + 1j * coeff_rand[0, 2],
                    coeff_rand[0, 3],
                    -coeff_rand[0, 1] + 1j * coeff_rand[0, 2],
                    np.nan,
                    np.nan,
                ],
                [
                    coeff_rand[0, 4] + 1j * coeff_rand[0, 5],
                    coeff_rand[0, 6] + 1j * coeff_rand[0, 7],
                    coeff_rand[0, 8],
                    -coeff_rand[0, 6] + 1j * coeff_rand[0, 7],
                    coeff_rand[0, 4] - 1j * coeff_rand[0, 5],
                ],
            ]
        )
        # First initialize and overwrite afterward to prevent normalization
        shd = SphericalHarmonicsDistributionComplex(
            np.array([[1, np.nan, np.nan], [0, 0, 0]])
        )
        shd.coeff_mat = unnormalized_coeffs
        shd.transformation = transformation
        int_val_num = shd.integrate_numerically()
        int_val_ana = shd.integrate()
        self.assertAlmostEqual(int_val_ana, int_val_num, places=5)

    def test_truncation(self):
        shd = SphericalHarmonicsDistributionComplex(self.unnormalized_coeffs)

        with self.assertWarns(UserWarning):
            shd2 = shd.truncate(4)
        self.assertEqual(shd2.coeff_mat.shape, (5, 9))
        self.assertTrue(
            np.all(np.isnan(shd2.coeff_mat[4, :]) | (shd2.coeff_mat[4, :] == 0))
        )
        shd3 = shd.truncate(5)
        self.assertEqual(shd3.coeff_mat.shape, (6, 11))
        self.assertTrue(
            np.all(
                np.isnan(shd3.coeff_mat[5:6, :]) | (shd3.coeff_mat[5:6, :] == 0),
                axis=(0, 1),
            )
        )
        shd4 = shd2.truncate(3)
        self.assertEqual(shd4.coeff_mat.shape, (4, 7))
        shd5 = shd3.truncate(3)
        self.assertEqual(shd5.coeff_mat.shape, (4, 7))

        phi, theta = np.random.rand(10) * 2 * np.pi, np.random.rand(10) * np.pi
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi, theta)
        self.assertTrue(
            np.allclose(
                shd2.pdf(np.column_stack((x, y, z))),
                shd.pdf(np.column_stack((x, y, z))),
                atol=1e-6,
            )
        )
        self.assertTrue(
            np.allclose(
                shd3.pdf(np.column_stack((x, y, z))),
                shd.pdf(np.column_stack((x, y, z))),
                atol=1e-6,
            )
        )
        self.assertTrue(
            np.allclose(
                shd4.pdf(np.column_stack((x, y, z))),
                shd.pdf(np.column_stack((x, y, z))),
                atol=1e-6,
            )
        )
        self.assertTrue(
            np.allclose(
                shd5.pdf(np.column_stack((x, y, z))),
                shd.pdf(np.column_stack((x, y, z))),
                atol=1e-6,
            )
        )

    @parameterized.expand(
        [
            # First, the basis functions that only yield real values are tested
            (
                "testl0m0",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda _, _1, z: np.ones_like(z) * np.sqrt(1 / (4 * np.pi)),
            ),
            (
                "testl1m0",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 1, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda _, _1, z: np.sqrt(3 / (4 * np.pi)) * z,
            ),
            (
                "testl2m0",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 1, 0, 0],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * np.sqrt(5 / np.pi)
                * (2 * z**2 - x**2 - y**2),
            ),
            (
                "testl3m0",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 1, 0, 0, 0],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * np.sqrt(7 / np.pi)
                * (z * (2 * z**2 - 3 * x**2 - 3 * y**2)),
            ),
            # For the other basis functions, complex values would be obtained.
            # Hence, combinations of complex basis function are used that are equal
            # to complex basis functions
            (
                "test_l1mneg1real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [1j * np.sqrt(1 / 2), 0, 1j * np.sqrt(1 / 2), np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda _, y, _1: np.sqrt(3 / (4 * np.pi)) * y,
            ),
            (
                "test_l1m1real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [np.sqrt(1 / 2), 0, -np.sqrt(1 / 2), np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda x, _, _1: np.sqrt(3 / (4 * np.pi)) * x,
            ),
            (
                "test_l2mneg2real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [1j * np.sqrt(1 / 2), 0, 0, 0, -1j * np.sqrt(1 / 2)],
                    ]
                ),
                lambda x, y, _: 1 / 2 * np.sqrt(15 / np.pi) * x * y,
            ),
            (
                "test_l2mneg1real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 1j * np.sqrt(1 / 2), 0, 1j * np.sqrt(1 / 2), 0],
                    ]
                ),
                lambda _, y, z: 1 / 2 * np.sqrt(15 / np.pi) * y * z,
            ),
            (
                "test_l2m1real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, np.sqrt(1 / 2), 0, -np.sqrt(1 / 2), 0],
                    ]
                ),
                lambda x, _, z: 1 / 2 * np.sqrt(15 / np.pi) * x * z,
            ),
            (
                "test_l2m2real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [np.sqrt(1 / 2), 0, 0, 0, np.sqrt(1 / 2)],
                    ]
                ),
                lambda x, y, _: 1 / 4 * np.sqrt(15 / np.pi) * (x**2 - y**2),
            ),
            (
                "test_l3mneg3real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [1j / np.sqrt(2), 0, 0, 0, 0, 0, 1j / np.sqrt(2)],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * np.sqrt(35 / (2 * np.pi))
                * y
                * (3 * x**2 - y**2),
            ),
            (
                "test_l3mneg2real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 1j / np.sqrt(2), 0, 0, 0, -1j / np.sqrt(2), 0],
                    ]
                ),
                lambda x, y, z: 1 / 2 * np.sqrt(105 / np.pi) * x * y * z,
            ),
            (
                "test_l3mneg1real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 0, 1j / np.sqrt(2), 0, 1j / np.sqrt(2), 0, 0],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * np.sqrt(21 / (2 * np.pi))
                * y
                * (4 * z**2 - x**2 - y**2),
            ),
            (
                "test_l3m1real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 0, 1 / np.sqrt(2), 0, -1 / np.sqrt(2), 0, 0],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * np.sqrt(21 / (2 * np.pi))
                * x
                * (4 * z**2 - x**2 - y**2),
            ),
            (
                "test_l3m2real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2), 0],
                    ]
                ),
                lambda x, y, z: 1 / 4 * np.sqrt(105 / np.pi) * z * (x**2 - y**2),
            ),
            (
                "test_l3m3real",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [1 / np.sqrt(2), 0, 0, 0, 0, 0, -1 / np.sqrt(2)],
                    ]
                ),
                lambda x, y, z: 1
                / 4
                * np.sqrt(35 / (2 * np.pi))
                * x
                * (x**2 - 3 * y**2),
            ),
        ]
    )
    def test_basis_function(self, _, coeff_mat, expected_func):
        shd = SphericalHarmonicsDistributionComplex(1 / np.sqrt(4 * np.pi))
        shd.coeff_mat = coeff_mat
        phi, theta = np.meshgrid(
            np.linspace(0, 2 * np.pi, 10), np.linspace(0, np.pi, 10)
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())
        np.testing.assert_allclose(
            shd.pdf(np.column_stack([x, y, z])), expected_func(x, y, z), atol=1e-6
        )

    @parameterized.expand(
        [
            # Test complex basis functions
            (
                "testl1mneg1_cart",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [1, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda x, y, _: 0.5 * np.sqrt(3 / (2 * np.pi)) * (x - 1j * y),
            ),
            (
                "testl1m1_cart",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 1, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda x, y, _: -0.5 * np.sqrt(3 / (2 * np.pi)) * (x + 1j * y),
            ),
            (
                "testl2mneg2_cart",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [1, 0, 0, 0, 0],
                    ]
                ),
                lambda x, y, _: 0.25 * np.sqrt(15 / (2 * np.pi)) * (x - 1j * y) ** 2,
            ),
            (
                "testl2mneg1_cart",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 1, 0, 0, 0],
                    ]
                ),
                lambda x, y, z: 0.5 * np.sqrt(15 / (2 * np.pi)) * (x - 1j * y) * z,
            ),
            (
                "testl2m1_cart",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 1, 0],
                    ]
                ),
                lambda x, y, z: -0.5 * np.sqrt(15 / (2 * np.pi)) * (x + 1j * y) * z,
            ),
            (
                "testl2m2_cart",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                lambda x, y, _: 0.25 * np.sqrt(15 / (2 * np.pi)) * (x + 1j * y) ** 2,
            ),
            # For spherical coordinates
            (
                "testl1mneg1_sph",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [1, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda phi, theta: 0.5
                * np.sqrt(3 / (2 * np.pi))
                * np.sin(theta)
                * np.exp(-1j * phi),
            ),
            (
                "testl1m1_sph",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 1, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda phi, theta: -0.5
                * np.sqrt(3 / (2 * np.pi))
                * np.sin(theta)
                * np.exp(1j * phi),
            ),
            (
                "testl2mneg2_sph",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [1, 0, 0, 0, 0],
                    ]
                ),
                lambda phi, theta: 0.25
                * np.sqrt(15 / (2 * np.pi))
                * np.sin(theta) ** 2
                * np.exp(-2j * phi),
            ),
            (
                "testl2mneg1_sph",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 1, 0, 0, 0],
                    ]
                ),
                lambda phi, theta: 0.5
                * np.sqrt(15 / (2 * np.pi))
                * np.sin(theta)
                * np.cos(theta)
                * np.exp(-1j * phi),
            ),
            (
                "testl2m1_sph",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 1, 0],
                    ]
                ),
                lambda phi, theta: -0.5
                * np.sqrt(15 / (2 * np.pi))
                * np.sin(theta)
                * np.cos(theta)
                * np.exp(1j * phi),
            ),
            (
                "testl2m2_sph",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                lambda phi, theta: 0.25
                * np.sqrt(15 / (2 * np.pi))
                * np.sin(theta) ** 2
                * np.exp(2j * phi),
            ),
            (
                "testl1mneg1_sphconv_colatitude",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [1, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda phi, theta: 0.5
                * np.sqrt(3 / (2 * np.pi))
                * np.sin(theta)
                * np.exp(-1j * phi),
            ),
            (
                "testl1m1_sphconv_colatitude",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 1, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
                lambda phi, theta: -0.5
                * np.sqrt(3 / (2 * np.pi))
                * np.sin(theta)
                * np.exp(1j * phi),
            ),
            (
                "testl2mneg2_sphconv_colatitude",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [1, 0, 0, 0, 0],
                    ]
                ),
                lambda phi, theta: 0.25
                * np.sqrt(15 / (2 * np.pi))
                * np.sin(theta) ** 2
                * np.exp(-2j * phi),
            ),
            (
                "testl2mneg1_sphconv_colatitude",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 1, 0, 0, 0],
                    ]
                ),
                lambda phi, theta: 0.5
                * np.sqrt(15 / (2 * np.pi))
                * np.sin(theta)
                * np.cos(theta)
                * np.exp(-1j * phi),
            ),
            (
                "testl2m1_sphconv_colatitude",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 1, 0],
                    ]
                ),
                lambda phi, theta: -0.5
                * np.sqrt(15 / (2 * np.pi))
                * np.sin(theta)
                * np.cos(theta)
                * np.exp(1j * phi),
            ),
            (
                "testl2m2_sphconv_colatitude",
                np.array(
                    [
                        [0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 1],
                    ]
                ),
                lambda phi, theta: 0.25
                * np.sqrt(15 / (2 * np.pi))
                * np.sin(theta) ** 2
                * np.exp(2j * phi),
            ),
        ]
    )
    def test_basis_function_complex(self, name, coeff_mat, expected_func):
        shd = SphericalHarmonicsDistributionComplex(
            1 / np.sqrt(4 * np.pi), assert_real=False
        )
        shd.coeff_mat = coeff_mat
        phi, theta = np.meshgrid(
            np.linspace(0, 2 * np.pi, 10), np.linspace(-np.pi / 2, np.pi / 2, 10)
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())

        vals_to_test = shd.pdf(np.column_stack([x, y, z]))
        if name.endswith("cart"):
            expected_func_vals = expected_func(x, y, z)
        elif name.endswith("sph"):
            expected_func_vals = expected_func(phi.ravel(), theta.ravel())
        elif name.endswith("colatitude"):
            phi, theta = AbstractSphericalDistribution.cart_to_sph(
                x, y, z, mode="colatitude"
            )
            expected_func_vals = expected_func(phi, theta)

        np.testing.assert_allclose(vals_to_test, expected_func_vals, atol=1e-6)

    @parameterized.expand(
        [
            (
                "l0m0",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l1mneg1",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [1j * np.sqrt(1 / 2), 0, 1j * np.sqrt(1 / 2), np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l1m0",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [0, 1, 0, np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l1m1",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [np.sqrt(1 / 2), 0, -np.sqrt(1 / 2), np.nan, np.nan],
                        [0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l2mneg2",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [1j * np.sqrt(1 / 2), 0, 0, 0, -1j * np.sqrt(1 / 2)],
                    ]
                ),
            ),
            (
                "l2mneg1",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 1j * np.sqrt(1 / 2), 0, 1j * np.sqrt(1 / 2), 0],
                    ]
                ),
            ),
            (
                "l2m0",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, 0, 1, 0, 0],
                    ]
                ),
            ),
            (
                "l2m1",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [0, np.sqrt(1 / 2), 0, -np.sqrt(1 / 2), 0],
                    ]
                ),
            ),
            (
                "l2m2",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan],
                        [np.sqrt(1 / 2), 0, 0, 0, np.sqrt(1 / 2)],
                    ]
                ),
            ),
            (
                "l3mneg3",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [1j / np.sqrt(2), 0, 0, 0, 0, 0, 1j / np.sqrt(2)],
                    ]
                ),
            ),
            (
                "l3mneg2",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 1j / np.sqrt(2), 0, 0, 0, -1j / np.sqrt(2), 0],
                    ]
                ),
            ),
            (
                "l3mneg1",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 0, 1j / np.sqrt(2), 0, 1j / np.sqrt(2), 0, 0],
                    ]
                ),
            ),
            (
                "l3m0",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 0, 0, 1, 0, 0, 0],
                    ]
                ),
            ),
            (
                "l3m1",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 0, 1 / np.sqrt(2), 0, -1 / np.sqrt(2), 0, 0],
                    ]
                ),
            ),
            (
                "l3m2",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [0, 1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2), 0],
                    ]
                ),
            ),
            (
                "l3m3",
                np.array(
                    [
                        [1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, np.nan, np.nan, np.nan, np.nan],
                        [0, 0, 0, 0, 0, np.nan, np.nan],
                        [1 / np.sqrt(2), 0, 0, 0, 0, 0, -1 / np.sqrt(2)],
                    ]
                ),
            ),
        ]
    )
    def test_conversion(self, _, coeff_mat):
        shd = SphericalHarmonicsDistributionComplex(coeff_mat)
        rshd = shd.to_spherical_harmonics_distribution_real()
        phi, theta = np.meshgrid(
            np.linspace(0, 2 * np.pi, 10), np.linspace(-np.pi / 2, np.pi / 2, 10)
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())
        np.testing.assert_allclose(
            rshd.pdf(np.column_stack((x, y, z))),
            shd.pdf(np.column_stack((x, y, z))),
            atol=1e-6,
        )

    @parameterized.expand(
        [
            (
                "shd_x",
                np.array([[1, np.nan, np.nan], [np.sqrt(1 / 2), 0, -np.sqrt(1 / 2)]]),
                np.array([1, 0, 0]),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_y",
                np.array(
                    [[1, np.nan, np.nan], [1j * np.sqrt(1 / 2), 0, 1j * np.sqrt(1 / 2)]]
                ),
                np.array([0, 1, 0]),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_z",
                np.array([[1, np.nan, np.nan], [0, 1, 0]]),
                np.array([0, 0, 1]),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_xy",
                np.array(
                    [
                        [1, np.nan, np.nan],
                        [
                            np.sqrt(1 / 2) + 1j * np.sqrt(1 / 2),
                            0,
                            -np.sqrt(1 / 2) + 1j * np.sqrt(1 / 2),
                        ],
                    ]
                ),
                np.array([1, 1, 0] / np.sqrt(2)),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_xz",
                np.array([[1, np.nan, np.nan], [np.sqrt(1 / 2), 1, -np.sqrt(1 / 2)]]),
                np.array([1, 0, 1] / np.sqrt(2)),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "shd_yz",
                np.array(
                    [[1, np.nan, np.nan], [1j * np.sqrt(1 / 2), 1, 1j * np.sqrt(1 / 2)]]
                ),
                np.array([0, 1, 1] / np.sqrt(2)),
                SphericalHarmonicsDistributionComplex.mean_direction,
            ),
            (
                "numerical_shd_x",
                np.array([[1, np.nan, np.nan], [np.sqrt(1 / 2), 0, -np.sqrt(1 / 2)]]),
                np.array([1, 0, 0]),
                SphericalHarmonicsDistributionComplex.mean_direction_numerical,
            ),
            (
                "numerical_shd_y",
                np.array(
                    [[1, np.nan, np.nan], [1j * np.sqrt(1 / 2), 0, 1j * np.sqrt(1 / 2)]]
                ),
                np.array([0, 1, 0]),
                SphericalHarmonicsDistributionComplex.mean_direction_numerical,
            ),
            (
                "numerical_shd_z",
                np.array([[1, np.nan, np.nan], [0, 1, 0]]),
                np.array([0, 0, 1]),
                SphericalHarmonicsDistributionComplex.mean_direction_numerical,
            ),
        ]
    )
    def test_mean_direction(self, _, input_array, expected_output, fun_to_test):
        shd = SphericalHarmonicsDistributionComplex(input_array)
        np.testing.assert_allclose(fun_to_test(shd), expected_output, atol=1e-10)

    def test_from_distribution_via_integral_vmf(self):
        # Test approximating a VMF
        dist = VonMisesFisherDistribution(np.array([-1, -1, 0] / np.sqrt(2)), 1)
        shd = SphericalHarmonicsDistributionComplex.from_distribution_via_integral(
            dist, 3
        )
        phi, theta = np.meshgrid(
            np.linspace(0, 2 * np.pi, 10), np.linspace(-np.pi / 2, np.pi / 2, 10)
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi.ravel(), theta.ravel())
        np.testing.assert_allclose(
            shd.mean_direction(), dist.mean_direction(), atol=1e-10
        )
        np.testing.assert_allclose(
            shd.mean_direction_numerical(), dist.mean_direction(), atol=1e-10
        )
        np.testing.assert_allclose(
            shd.integrate_numerically(), dist.integrate_numerically(), atol=1e-10
        )
        np.testing.assert_allclose(
            shd.pdf(np.column_stack([x, y, z])),
            dist.pdf(np.column_stack([x, y, z])),
            atol=0.001,
        )

    def test_from_distribution_via_integral_uniform(self):
        shd = SphericalHarmonicsDistributionComplex.from_distribution_via_integral(
            HypersphericalUniformDistribution(2), degree=0
        )
        np.testing.assert_allclose(shd.coeff_mat, np.array([[1 / np.sqrt(4 * np.pi)]]))

    def test_transformation_via_integral_shd(self):
        # Test approximating a spherical harmonic distribution
        dist = SphericalHarmonicsDistributionComplex(
            np.array([[1, np.nan, np.nan], [0, 1, 0]])
        )

        shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_cart(
            dist.pdf, 1
        )
        np.testing.assert_allclose(shd.coeff_mat, dist.coeff_mat, atol=1e-6)

    def test_convergence(self):
        no_diffs = 3
        dist = VonMisesFisherDistribution(np.array([0, -1, 0]), 1)
        diffs = np.zeros(no_diffs)

        for i in range(0, no_diffs):
            shd = SphericalHarmonicsDistributionComplex.from_function_via_integral_cart(
                dist.pdf, i
            )
            diffs[i] = shd.total_variation_distance_numerical(dist)

        # Check if the deviation from true density is decreasing
        self.assertTrue(np.all(np.diff(diffs) < 0))

    @parameterized.expand(
        [
            ("zplus", [[1 / np.sqrt(4 * np.pi), np.nan, np.nan], [0, 1, 0]], [0, 0, 1]),
            (
                "zminus",
                [[1 / np.sqrt(4 * np.pi), np.nan, np.nan], [0, -1, 0]],
                [0, 0, -1],
            ),
            (
                "yplus",
                [
                    [1 / np.sqrt(4 * np.pi), np.nan, np.nan],
                    [1j * np.sqrt(1 / 2), 0, 1j * np.sqrt(1 / 2)],
                ],
                [0, 1, 0],
            ),
            (
                "yminus",
                [
                    [1 / np.sqrt(4 * np.pi), np.nan, np.nan],
                    [-1j * np.sqrt(1 / 2), 0, -1j * np.sqrt(1 / 2)],
                ],
                [0, -1, 0],
            ),
            (
                "xplus",
                [
                    [1 / np.sqrt(4 * np.pi), np.nan, np.nan],
                    [np.sqrt(1 / 2), 0, -np.sqrt(1 / 2)],
                ],
                [1, 0, 0],
            ),
            (
                "xminus",
                [
                    [1 / np.sqrt(4 * np.pi), np.nan, np.nan],
                    [-np.sqrt(1 / 2), 0, np.sqrt(1 / 2)],
                ],
                [-1, 0, 0],
            ),
            (
                "xyplus",
                [
                    [1 / np.sqrt(4 * np.pi), np.nan, np.nan],
                    [
                        1j * np.sqrt(1 / 2) + np.sqrt(1 / 2),
                        1,
                        1j * np.sqrt(1 / 2) - np.sqrt(1 / 2),
                    ],
                ],
                1 / np.sqrt(3) * np.array([1, 1, 1]),
            ),
            (
                "xyminus",
                [
                    [1 / np.sqrt(4 * np.pi), np.nan, np.nan],
                    [
                        -1j * np.sqrt(1 / 2) - np.sqrt(1 / 2),
                        0,
                        -1j * np.sqrt(1 / 2) + np.sqrt(1 / 2),
                    ],
                ],
                1 / np.sqrt(2) * np.array([-1, -1, 0]),
            ),
        ]
    )
    def test_mean(self, _, coeff_mat, expected_output):
        shd = SphericalHarmonicsDistributionComplex(coeff_mat)
        np.testing.assert_allclose(shd.mean_direction(), expected_output, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
