import copy
import unittest

import numpy as np
from pyrecest.distributions import HypertoroidalWDDistribution, ToroidalWDDistribution


class TestHypertoroidalWDDistribution(unittest.TestCase):
    def test_hypertoroidal_wd_distribution(self):
        d = np.array(
            [[0.5, 2, 0.5], [3, 2, 0.2], [4, 5, 5.8], [6, 3, 4.3], [6, 0, 1.2]]
        )
        w = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
        twd = HypertoroidalWDDistribution(d, w)

        # Test class, dimensions, and weights
        self.assertIsInstance(twd, HypertoroidalWDDistribution)
        np.testing.assert_array_almost_equal(twd.d, d)
        np.testing.assert_array_almost_equal(twd.w, w)

        # Test trigonometric moment
        m = twd.trigonometric_moment(1)
        m1 = twd.marginalize_to_1D(0).trigonometric_moment(1)
        m2 = twd.marginalize_to_1D(1).trigonometric_moment(1)
        np.testing.assert_almost_equal(m[0], m1, decimal=10)
        np.testing.assert_almost_equal(m[1], m2, decimal=10)
        np.testing.assert_almost_equal(
            m[0], np.sum(w * np.exp(1j * d[:, 0])), decimal=10
        )
        np.testing.assert_almost_equal(
            m[1], np.sum(w * np.exp(1j * d[:, 1])), decimal=10
        )

        # Test sampling
        n_samples = 5
        s = twd.sample(n_samples)
        self.assertEqual(s.shape, (n_samples, d.shape[-1]))
        np.testing.assert_array_almost_equal(s, np.mod(s, 2 * np.pi))

        # Test getMarginal
        for i in range(d.shape[-1]):
            wd = twd.marginalize_to_1D(i)
            np.testing.assert_array_almost_equal(twd.w, wd.w)
            np.testing.assert_array_almost_equal(np.squeeze(wd.d), twd.d[:, i])

        # Test apply function
        same = twd.apply_function(lambda x: x)
        np.testing.assert_array_almost_equal(
            same.trigonometric_moment(1), twd.trigonometric_moment(1)
        )
        shift_offset = np.array([1.4, -0.3, np.pi])
        shifted = twd.apply_function(lambda x: x + shift_offset)
        np.testing.assert_almost_equal(
            shifted.trigonometric_moment(1),
            twd.trigonometric_moment(1) * np.exp(1j * shift_offset),
            decimal=10,
        )

    def test_shift(self):
        d = np.array(
            [
                [4, -2, 0.01],
                [3, 2, 0],
                [2.5, 4, -2.4],
                [8, 1 / 3, 2.2],
                [2.1, 7 / 99, 0.2],
            ]
        )

        w = np.array([0.3, 0.3, 0.3, 0.05, 0.05])
        twd = HypertoroidalWDDistribution(d, w)
        s = np.array([1, -3, 6])
        twd_shifted = twd.shift(s)
        self.assertIsInstance(twd_shifted, HypertoroidalWDDistribution)
        np.testing.assert_array_almost_equal(twd.w, twd_shifted.w)
        np.testing.assert_array_almost_equal(
            twd.d,
            np.mod(twd_shifted.d - np.outer(np.ones_like(w), s), 2 * np.pi),
            decimal=10,
        )

    @staticmethod
    def get_pseudorandom_hypertoroidal_wd(dim=2):
        np.random.seed(0)
        n = 20
        d = 2 * np.pi * np.random.rand(n, dim)
        w = np.random.rand(n)
        w = w / np.sum(w)
        hwd = HypertoroidalWDDistribution(d, w)
        return hwd

    def test_to_toroidal_wd(self):
        hwd = TestHypertoroidalWDDistribution.get_pseudorandom_hypertoroidal_wd(2)
        twd1 = ToroidalWDDistribution(copy.copy(hwd.d), copy.copy(hwd.w))
        twd2 = hwd.to_toroidal_wd()
        self.assertIsInstance(twd2, ToroidalWDDistribution)
        np.testing.assert_array_almost_equal(twd1.d, twd2.d, decimal=10)
        np.testing.assert_array_almost_equal(twd1.w, twd2.w, decimal=10)

    def test_marginalization(self):
        hwd = TestHypertoroidalWDDistribution.get_pseudorandom_hypertoroidal_wd(2)
        wd1 = hwd.marginalize_to_1D(0)
        wd2 = hwd.marginalize_out(1)
        np.testing.assert_array_almost_equal(wd1.d, np.squeeze(wd2.d))
        np.testing.assert_array_almost_equal(wd1.w, wd2.w)


if __name__ == "__main__":
    unittest.main()
