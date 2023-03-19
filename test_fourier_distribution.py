import unittest
import numpy as np
from fourier_distribution import FourierDistribution
from vm_distribution import VMDistribution
import copy
import scipy.integrate as integrate

class TestFourierDistribution(unittest.TestCase):
    def test_vm_to_fourier(self):
        for mult_by_n in [True, False]:
            for transformation in ['identity', 'sqrt']:
                xs = np.linspace(0, 2*np.pi, 100)
                dist = VMDistribution(np.array(2.5, dtype=np.float32), np.array(1.5, dtype=np.float32))
                fd = FourierDistribution.from_distribution(dist, n=31, transformation=transformation, store_values_multiplied_by_n=mult_by_n)
                np.testing.assert_array_almost_equal(dist.pdf(xs), fd.pdf(xs))
                fd_real = fd.to_real_fd()
                np.testing.assert_array_almost_equal(dist.pdf(xs), fd_real.pdf(xs))

    def test_integral_numerical(self):
        scale_by = 2/5
        for mult_by_n in [True, False]:
            for transformation in ['identity']:  # ,'sqrt']:
                dist = VMDistribution(np.array(1.5, dtype=np.float32), np.array(2.5, dtype=np.float32))
                fd = FourierDistribution.from_distribution(dist, n=31, transformation=transformation, store_values_multiplied_by_n=mult_by_n)
                np.testing.assert_array_almost_equal(fd.integral_numerical(), 1)
                fd_real = fd.to_real_fd()
                np.testing.assert_array_almost_equal(fd_real.integral_numerical(), 1)
                fd_unnorm = copy.copy(fd)
                fd_unnorm.c = fd.c * (scale_by)
                if transformation == 'identity':
                    expected_val = scale_by
                else:
                    expected_val = (scale_by)**2
                np.testing.assert_array_almost_equal(fd_unnorm.integral_numerical(), expected_val)
                fd_unnorm_real = fd_unnorm.to_real_fd()
                np.testing.assert_array_almost_equal(fd_unnorm_real.integral_numerical(), expected_val)

    def test_integral(self):
        scale_by = 1/5
        for mult_by_n in [True, False]:
            for transformation in ['identity', 'sqrt']:
                dist = VMDistribution(np.array(2.5, dtype=np.float32), np.array(1.5, dtype=np.float32))
                fd = FourierDistribution.from_distribution(dist, n=31, transformation=transformation, store_values_multiplied_by_n=mult_by_n)
                np.testing.assert_array_almost_equal(fd.integral(), 1)
                fd_real = fd.to_real_fd()
                np.testing.assert_array_almost_equal(fd_real.integral(), 1)
                fd_unnorm = copy.copy(fd)
                fd_unnorm.c = fd.c * (scale_by)
                if transformation == 'identity':
                    expected_val = scale_by
                else:
                    expected_val = (scale_by)**2
                np.testing.assert_array_almost_equal(fd_unnorm.integral(), expected_val)
                fd_unnorm_real = fd_unnorm.to_real_fd()
                np.testing.assert_array_almost_equal(fd_unnorm_real.integral(), expected_val)
                fd_unnorm = FourierDistribution.from_distribution(dist, n=31, transformation=transformation, store_values_multiplied_by_n=mult_by_n)
                fd_unnorm.c = fd_unnorm.c * scale_by
                fd_norm = fd_unnorm.normalize()
                fd_unnorm_real = fd_unnorm.to_real_fd()
                fd_norm_real = fd_unnorm_real.normalize()
                np.testing.assert_array_almost_equal(fd_norm.integral(), 1)
                np.testing.assert_array_almost_equal(fd_norm_real.integral(), 1)

    def test_distance(self):
        dist1 = VMDistribution(np.array(0.0, dtype=np.float32), np.array(1.0, dtype=np.float32))
        dist2 = VMDistribution(np.array(2.0, dtype=np.float32), np.array(1.0, dtype=np.float32))
        for mult_by_n in [False, True]:
            fd1 = FourierDistribution.from_distribution(dist1, n=31, transformation='sqrt', store_values_multiplied_by_n=mult_by_n)
            fd2 = FourierDistribution.from_distribution(dist2, n=31, transformation='sqrt', store_values_multiplied_by_n=mult_by_n)
            hel_like_distance, _ = integrate.quad(lambda x: (np.sqrt(dist1.pdf(np.array(x).reshape(1, -1))) - np.sqrt(dist2.pdf(np.array(x).reshape(1, -1))))**2, 0, 2*np.pi)
            fd_diff = fd1 - fd2
            np.testing.assert_array_almost_equal(fd_diff.integral(), hel_like_distance)

if __name__ == "__main__":
    unittest.main()
