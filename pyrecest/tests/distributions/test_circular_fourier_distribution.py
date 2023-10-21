from math import pi
from pyrecest.backend import sqrt
from pyrecest.backend import linspace
from pyrecest.backend import ceil
from pyrecest.backend import array
from pyrecest.backend import arange
from pyrecest.backend import allclose
from pyrecest.backend import all
import pyrecest.backend
import copy
import unittest


from parameterized import parameterized
from pyrecest.distributions import (
    CircularFourierDistribution,
    VonMisesDistribution,
    WrappedNormalDistribution,
)
from scipy import integrate


class TestCircularFourierDistribution(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "identity",
                VonMisesDistribution,
                array(0.4),
                arange(0.1, 2.1, 0.1),
                101,
            ),
            ("sqrt", VonMisesDistribution, 0.5, arange(0.1, 2.1, 0.1), 101, 1e-8),
            (
                "identity",
                WrappedNormalDistribution,
                array(0.8),
                arange(0.2, 2.1, 0.1),
                101,
            ),
            (
                "sqrt",
                WrappedNormalDistribution,
                array(0.8),
                arange(0.2, 2.1, 0.1),
                101,
            ),
        ]
    )
    # pylint: disable=too-many-arguments
    def test_fourier_conversion(
        self, transformation, dist_class, mu, param_range, coeffs, tolerance
    ):
        """
        Test fourier conversion of the given distribution with varying parameter.
        """
        for param in param_range:
            dist = dist_class(mu, param)
            xvals = arange(-2.0 * pi, 3.0 * pi, 0.01)
            fd = CircularFourierDistribution.from_distribution(
                dist, coeffs, transformation
            )
            self.assertEqual(
                fd.c.shape[0],
                ceil(coeffs / 2.0),
                "Length of Fourier Coefficients mismatch.",
            )
            self.assertTrue(
                allclose(fd.pdf(xvals), dist.pdf(xvals)),
                "PDF values do not match.",
            )

    @parameterized.expand(
        [
            (True, "identity"),
            (True, "sqrt"),
            (False, "identity"),
            (False, "sqrt"),
        ]
    )
    def test_vm_to_fourier(self, mult_by_n, transformation):
        xs = linspace(0.0, 2.0 * pi, 100)
        dist = VonMisesDistribution(2.5, 1.5)
        fd = CircularFourierDistribution.from_distribution(
            dist,
            n=31,
            transformation=transformation,
            store_values_multiplied_by_n=mult_by_n,
        )
        np.testing.assert_array_almost_equal(dist.pdf(xs), fd.pdf(xs))
        fd_real = fd.to_real_fd()
        np.testing.assert_array_almost_equal(dist.pdf(xs), fd_real.pdf(xs))

    @parameterized.expand(
        [
            (True, "identity"),
            (False, "identity"),
            (True, "sqrt"),
            (False, "sqrt"),
        ]
    )
    @unittest.skipIf(pyrecest.backend.__name__ == 'pyrecest.pytorch', reason="Not supported on PyTorch backend")
    def test_integrate_numerically(self, mult_by_n, transformation):
        scale_by = 2.0 / 5.0
        dist = VonMisesDistribution(2.9, 1.3)
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

    @parameterized.expand(
        [
            (True, "identity"),
            (False, "identity"),
            (True, "sqrt"),
            (False, "sqrt"),
        ]
    )
    def test_integrate(self, mult_by_n, transformation):
        scale_by = 1 / 5
        dist = VonMisesDistribution(2.9, 1.3)
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
        fd_unnorm.c = fd.c * scale_by
        if transformation == "identity":
            expected_val = scale_by
        else:
            expected_val = scale_by ** 2
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
        np.testing.assert_array_almost_equal(fd_norm.integrate(), 1.0)
        np.testing.assert_array_almost_equal(fd_norm_real.integrate(), 1.0)

    @parameterized.expand(
        [
            (False,),
            (True,),
        ]
    )
    def test_distance(self, mult_by_n):
        dist1 = VonMisesDistribution(array(0.0), array(1.0))
        dist2 = VonMisesDistribution(array(2.0), array(1.0))
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
                sqrt(dist1.pdf(array(x))) - sqrt(dist2.pdf(array(x)))
            )
            ** 2,
            0.0,
            2.0 * pi,
        )
        fd_diff = fd1 - fd2
        np.testing.assert_array_almost_equal(fd_diff.integrate(), hel_like_distance)


if __name__ == "__main__":
    unittest.main()