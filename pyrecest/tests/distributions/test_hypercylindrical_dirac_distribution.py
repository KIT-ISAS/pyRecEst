from math import pi
from pyrecest.backend import random
from pyrecest.backend import sum
from pyrecest.backend import ones_like
from pyrecest.backend import ones
from pyrecest.backend import isclose
from pyrecest.backend import eye
from pyrecest.backend import exp
from pyrecest.backend import array
from pyrecest.backend import zeros_like
from pyrecest.backend import zeros
import unittest

import numpy as np
from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import (
    HypercylindricalDiracDistribution,
)
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)
from scipy.stats import wishart


class TestHypercylindricalDiracDistribution(unittest.TestCase):
    def setUp(self):
        self.d = array(
            [[1, 2, 3, 4, 5, 6], [2, 4, 0, 0.5, 1, 1], [0, 10, 20, 30, 40, 50]]
        ).T
        self.w = array([1, 2, 3, 1, 2, 3])
        self.w = self.w / sum(self.w)
        self.pwd = HypercylindricalDiracDistribution(1, self.d, self.w)

    def test_mean_and_marginalization(self):
        mean = self.pwd.hybrid_moment()
        wd = self.pwd.marginalize_linear()
        assert isclose(mean[0], wd.trigonometric_moment(1).real, rtol=1e-10)
        assert isclose(mean[1], wd.trigonometric_moment(1).imag, rtol=1e-10)
        assert isclose(mean[2], sum(self.w * self.d[:, 1]), rtol=1e-10)
        assert isclose(mean[3], sum(self.w * self.d[:, 2]), rtol=1e-10)

    def test_covariance(self):
        clin = self.pwd.linear_covariance()
        assert clin.shape == (self.pwd.lin_dim, self.pwd.lin_dim)

    def test_apply_function_identity(self):
        same = self.pwd.apply_function(lambda x: x)
        np.testing.assert_array_equal(self.pwd.d, same.d)
        np.testing.assert_array_equal(self.pwd.w, same.w)
        assert self.pwd.lin_dim == same.lin_dim
        assert self.pwd.bound_dim == same.bound_dim

    def test_apply_function_shift(self):
        shift_offset = array([1.4, -0.3, 1])

        def shift(x, shift_by=shift_offset):
            return x + shift_by

        shifted = self.pwd.apply_function(shift)
        assert isclose(
            shifted.marginalize_linear().trigonometric_moment(1),
            self.pwd.marginalize_linear().trigonometric_moment(1)
            * exp(1j * shift_offset[0]),
            rtol=1e-10,
        )

    def test_reweigh(self):
        # Define functions for testing
        def f1(x):
            return sum(x, axis=-1) == 3

        def f2(x):
            return 2 * ones(x.shape[0])

        def f3(x):
            return x[:, 0]

        pwd_rew1 = self.pwd.reweigh(f1)
        pwd_rew2 = self.pwd.reweigh(f2)
        pwd_rew3 = self.pwd.reweigh(f3)

        assert isinstance(pwd_rew1, HypercylindricalDiracDistribution)
        np.testing.assert_array_equal(pwd_rew1.d, self.pwd.d)
        np.testing.assert_array_equal(pwd_rew1.w, f1(self.pwd.d))

        assert isinstance(pwd_rew2, HypercylindricalDiracDistribution)
        np.testing.assert_array_equal(pwd_rew2.d, self.pwd.d)
        np.testing.assert_array_equal(pwd_rew2.w, self.pwd.w)

        assert isinstance(pwd_rew3, HypercylindricalDiracDistribution)
        np.testing.assert_array_equal(pwd_rew3.d, self.pwd.d)
        w_new = self.pwd.d[:, 0] * self.pwd.w
        np.testing.assert_array_equal(pwd_rew3.w, w_new / sum(w_new))

    def test_sampling(self):
        random.seed(0)
        n = 10
        s = self.pwd.sample(n)
        assert s.shape == (n, 3)
        s = s[:, 0]
        self.assertTrue(all(s >= zeros_like(s)))
        self.assertTrue(all(s < 2 * pi * ones_like(s)))

    def test_from_distribution(self):
        random_gen = random.default_rng(0)  # Could fail randomly otherwise
        df = 4
        scale = eye(4)
        C = wishart.rvs(df, scale, random_state=random_gen)
        hwn = PartiallyWrappedNormalDistribution(array([1, 2, 3, 4]), C, 2)
        hddist = HypercylindricalDiracDistribution.from_distribution(hwn, 100000)
        np.testing.assert_allclose(hddist.hybrid_mean(), hwn.hybrid_mean(), atol=0.15)