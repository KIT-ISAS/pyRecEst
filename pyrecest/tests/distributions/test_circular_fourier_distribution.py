import copy
import unittest

import numpy as np
from pyrecest.distributions import (
    CircularFourierDistribution,
    VonMisesDistribution,
    WrappedNormalDistribution,
)
from scipy import integrate


class TestCircularFourierDistribution(unittest.TestCase):
    def _test_fourier_conversion(self, dist, coeffs, transformation, tolerance):
        xvals = np.arange(-2 * np.pi, 3 * np.pi, 0.01)
        fd = CircularFourierDistribution.from_distribution(dist, coeffs, transformation)
        self.assertEqual(len(fd.c), np.ceil(coeffs / 2))
        self.assertTrue(np.allclose(fd.pdf(xvals), dist.pdf(xvals), atol=tolerance))

    def test_vm_to_fourier_id(self):
        mu = 0.4
        for kappa in np.arange(0.1, 2.1, 0.1):
            dist = VonMisesDistribution(mu, kappa)
            self._test_fourier_conversion(dist, 101, "identity", 1e-8)

    def test_vm_to_fourier_sqrt(self):
        mu = 0.5
        for kappa in np.arange(0.1, 2.1, 0.1):
            dist = VonMisesDistribution(mu, kappa)
            self._test_fourier_conversion(dist, 101, "sqrt", 1e-8)

    def test_wn_to_fourier_id(self):
        mu = np.array(0.8)
        for sigma in np.arange(0.2, 2.1, 0.1):
            dist = WrappedNormalDistribution(mu, sigma)
            self._test_fourier_conversion(dist, 101, "identity", 1e-8)

    def test_wn_to_fourier_sqrt(self):
        mu = np.array(0.8)
        for sigma in np.arange(0.2, 2.1, 0.1):
            dist = WrappedNormalDistribution(mu, sigma)
            self._test_fourier_conversion(dist, 101, "sqrt", 1e-8)

    def test_vm_to_fourier(self):
        for mult_by_n in [True, False]:
            for transformation in ["identity", "sqrt"]:
                xs = np.linspace(0, 2 * np.pi, 100)
                dist = VonMisesDistribution(
                    np.array(2.5, dtype=np.float32), np.array(1.5, dtype=np.float32)
                )
                fd = CircularFourierDistribution.from_distribution(
                    dist,
                    n=31,
                    transformation=transformation,
                    store_values_multiplied_by_n=mult_by_n,
                )
                np.testing.assert_array_almost_equal(dist.pdf(xs), fd.pdf(xs))
                fd_real = fd.to_real_fd()
                np.testing.assert_array_almost_equal(dist.pdf(xs), fd_real.pdf(xs))

    def _test_integrate_numerically_helper(self, mult_by_n, transformation):
        scale_by = 2 / 5
        dist = VonMisesDistribution(np.array(2.9), np.array(1.3))
        fd = CircularFourierDistribution.from_distribution(
            dist,
            n=31,
            transformation=transformation,
            store_values_multiplied_by_n=mult_by_n,
        )
        np.testing.assert_array_almost_equal(fd.integrate_numerically(), 1)
        fd_real = fd.to_real_fd()
        np.testing.assert_array_almost_equal(fd_real.integrate_numerically(), 1)
        fd_unnorm = copy.copy(fd)
        fd_unnorm.c = fd.c * (scale_by)
        if transformation == "identity":
            expected_val = scale_by
        else:
            expected_val = (scale_by) ** 2
        np.testing.assert_array_almost_equal(
            fd_unnorm.integrate_numerically(), expected_val
        )
        fd_unnorm_real = fd_unnorm.to_real_fd()
        np.testing.assert_array_almost_equal(
            fd_unnorm_real.integrate_numerically(), expected_val
        )

    def test_integrate_numerically_identity_mult_true(self):
        self._test_integrate_numerically_helper(
            mult_by_n=True, transformation="identity"
        )

    def test_integrate_numerically_identity_mult_false(self):
        self._test_integrate_numerically_helper(
            mult_by_n=False, transformation="identity"
        )

    def test_integrate_numerically_sqrt_mult_true(self):
        self._test_integrate_numerically_helper(mult_by_n=True, transformation="sqrt")

    def test_integrate_numerically_sqrt_mult_false(self):
        self._test_integrate_numerically_helper(mult_by_n=False, transformation="sqrt")

    def _test_integrate_helper(self, mult_by_n, transformation):
        scale_by = 1 / 5
        dist = VonMisesDistribution(np.array(2.9), np.array(1.3))
        fd = CircularFourierDistribution.from_distribution(
            dist,
            n=31,
            transformation=transformation,
            store_values_multiplied_by_n=mult_by_n,
        )
        np.testing.assert_array_almost_equal(fd.integrate(), 1)
        fd_real = fd.to_real_fd()
        np.testing.assert_array_almost_equal(fd_real.integrate(), 1)
        fd_unnorm = copy.copy(fd)
        fd_unnorm.c = fd.c * (scale_by)
        if transformation == "identity":
            expected_val = scale_by
        else:
            expected_val = (scale_by) ** 2
        np.testing.assert_array_almost_equal(fd_unnorm.integrate(), expected_val)
        fd_unnorm_real = fd_unnorm.to_real_fd()
        np.testing.assert_array_almost_equal(fd_unnorm_real.integrate(), expected_val)
        fd_unnorm = CircularFourierDistribution.from_distribution(
            dist,
            n=31,
            transformation=transformation,
            store_values_multiplied_by_n=mult_by_n,
        )
        fd_unnorm.c = fd_unnorm.c * scale_by
        fd_norm = fd_unnorm.normalize()
        fd_unnorm_real = fd_unnorm.to_real_fd()
        fd_norm_real = fd_unnorm_real.normalize()
        np.testing.assert_array_almost_equal(fd_norm.integrate(), 1)
        np.testing.assert_array_almost_equal(fd_norm_real.integrate(), 1)

    def test_integrate_identity_mult_true(self):
        self._test_integrate_helper(mult_by_n=True, transformation="identity")

    def test_integrate_identity_mult_false(self):
        self._test_integrate_helper(mult_by_n=False, transformation="identity")

    def test_integrate_sqrt_mult_true(self):
        self._test_integrate_helper(mult_by_n=True, transformation="sqrt")

    def test_integrate_sqrt_mult_false(self):
        self._test_integrate_helper(mult_by_n=False, transformation="sqrt")

    def test_distance(self):
        dist1 = VonMisesDistribution(np.array(0.0), np.array(1.0))
        dist2 = VonMisesDistribution(np.array(2.0), np.array(1.0))
        for mult_by_n in [False, True]:
            fd1 = CircularFourierDistribution.from_distribution(
                dist1,
                n=31,
                transformation="sqrt",
                store_values_multiplied_by_n=mult_by_n,
            )
            fd2 = CircularFourierDistribution.from_distribution(
                dist2,
                n=31,
                transformation="sqrt",
                store_values_multiplied_by_n=mult_by_n,
            )
            hel_like_distance, _ = integrate.quad(
                lambda x: (
                    np.sqrt(dist1.pdf(np.array(x).reshape(1, -1)))
                    - np.sqrt(dist2.pdf(np.array(x).reshape(1, -1)))
                )
                ** 2,
                0,
                2 * np.pi,
            )
            fd_diff = fd1 - fd2
            np.testing.assert_array_almost_equal(fd_diff.integrate(), hel_like_distance)


if __name__ == "__main__":
    unittest.main()
