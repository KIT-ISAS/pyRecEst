import unittest

import numpy as np
from pyrecest.distributions import VonMisesFisherDistribution

vectors_to_test_2d = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0] / np.sqrt(2),
        [1, 1, 2] / np.linalg.norm([1, 1, 2]),
        -np.array([1, 1, 2]) / np.linalg.norm([1, 1, 2]),
    ]
)


class TestVonMisesFisherDistribution(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([1, 2, 3])
        self.mu = self.mu / np.linalg.norm(self.mu)
        self.kappa = 2
        self.vmf = VonMisesFisherDistribution(self.mu, self.kappa)
        self.other = VonMisesFisherDistribution(np.array([0, 0, 1]), self.kappa / 3)

    def test_vmf_distribution_3d_sanity_check(self):
        self.assertIsInstance(self.vmf, VonMisesFisherDistribution)
        self.assertTrue(np.allclose(self.vmf.mu, self.mu))
        self.assertEqual(self.vmf.kappa, self.kappa)
        self.assertEqual(self.vmf.dim + 1, len(self.mu))

    def test_vmf_distribution_3d_mode(self):
        self.assertTrue(
            np.allclose(self.vmf.mode_numerical(), self.vmf.mode(), atol=1e-5)
        )

    def test_vmf_distribution_3d_integral(self):
        self.assertAlmostEqual(self.vmf.integrate(), 1, delta=1e-5)

    def test_vmf_distribution_3d_multiplication(self):
        vmf_mul = self.vmf.multiply(self.other)
        vmf_mul2 = self.other.multiply(self.vmf)
        c = vmf_mul.pdf(np.array([1, 0, 0])) / (
            self.vmf.pdf(np.array([1, 0, 0])) * self.other.pdf(np.array([1, 0, 0]))
        )
        x = np.array([0, 1, 0])
        self.assertAlmostEqual(
            self.vmf.pdf(x) * self.other.pdf(x) * c, vmf_mul.pdf(x), delta=1e-10
        )
        self.assertAlmostEqual(
            self.vmf.pdf(x) * self.other.pdf(x) * c, vmf_mul2.pdf(x), delta=1e-10
        )

    def test_vmf_distribution_3d_convolve(self):
        vmf_conv = self.vmf.convolve(self.other)
        self.assertTrue(np.allclose(vmf_conv.mu, self.vmf.mu, atol=1e-10))
        d = 3
        self.assertAlmostEqual(
            VonMisesFisherDistribution.a_d(d, vmf_conv.kappa),
            VonMisesFisherDistribution.a_d(d, self.vmf.kappa)
            * VonMisesFisherDistribution.a_d(d, self.other.kappa),
            delta=1e-10,
        )

    def test_init_2d(self):
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        dist = VonMisesFisherDistribution(mu, kappa)
        np.testing.assert_array_almost_equal(dist.C, 7.22562325261744e-05)

    def test_init_3d(self):
        mu = np.array([1, 1, 2, -3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        dist = VonMisesFisherDistribution(mu, kappa)
        np.testing.assert_array_almost_equal(dist.C, 0.0318492506152322)

    def test_pdf_2d(self):
        mu = np.array([1, 1, 2])
        mu = mu / np.linalg.norm(mu)
        kappa = 10
        dist = VonMisesFisherDistribution(mu, kappa)

        np.testing.assert_array_almost_equal(
            dist.pdf(vectors_to_test_2d),
            np.array(
                [
                    0.00428425301914546,
                    0.00428425301914546,
                    0.254024093013817,
                    0.0232421165060131,
                    1.59154943419939,
                    3.28042788159008e-09,
                ],
            ),
        )

    def test_pdf_3d(self):
        mu = np.array([1, 1, 2, -3])
        mu = mu / np.linalg.norm(mu)
        kappa = 2
        dist = VonMisesFisherDistribution(mu, kappa)

        xs_unnorm = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 1, 0, 0],
                [1, -1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, -1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, -1],
            ]
        )
        xs = xs_unnorm / np.linalg.norm(xs_unnorm, axis=1, keepdims=True)

        np.testing.assert_array_almost_equal(
            dist.pdf(xs),
            np.array(
                [
                    0.0533786916025838,
                    0.0533786916025838,
                    0.0894615936690536,
                    0.00676539409350726,
                    0.0661093769275549,
                    0.0318492506152322,
                    0.0952456142366906,
                    0.0221063629087443,
                    0.0153438863274034,
                    0.13722300001807,
                ],
            ),
        )

    def test_mean_direction(self):
        mu = 1 / np.sqrt(2) * np.array([1, 1, 0])
        vmf = VonMisesFisherDistribution(mu, 1)
        self.assertTrue(np.allclose(vmf.mean_direction(), mu, atol=1e-13))

    def _test_hellinger_distance_helper(
        self, dist1, dist2, delta=1e-10, numerical_delta=1e-10
    ):
        self.assertAlmostEqual(dist1.hellinger_distance(dist1), 0, delta=delta)
        self.assertAlmostEqual(dist2.hellinger_distance(dist2), 0, delta=delta)
        self.assertAlmostEqual(
            dist1.hellinger_distance(dist2),
            dist1.hellinger_distance_numerical(dist2),
            delta=numerical_delta,
        )
        self.assertAlmostEqual(
            dist1.hellinger_distance(dist2),
            dist2.hellinger_distance(dist1),
            delta=delta,
        )

    def test_hellinger_distance_2d(self):
        # 2D
        vmf1 = VonMisesFisherDistribution(np.array([1, 0]), 0.9)
        vmf2 = VonMisesFisherDistribution(np.array([0, 1]), 1.7)
        self._test_hellinger_distance_helper(vmf1, vmf2)

    def test_hellinger_distance_3d(self):
        # 3D
        vmf1 = VonMisesFisherDistribution(np.array([1, 0, 0]), 0.6)
        mu2 = np.array([1, 2, 3])
        vmf2 = VonMisesFisherDistribution(mu2 / np.linalg.norm(mu2), 2.1)
        self._test_hellinger_distance_helper(vmf1, vmf2, numerical_delta=1e-6)
      
    def test_from_distribution(self):
        v = VonMisesFisherDistribution.from_distribution(self.vmf)
        np.testing.assert_allclose(v.mu, self.vmf.mu, atol=1e-2)
        np.testing.assert_allclose(v.kappa, 0, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
