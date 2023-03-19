import unittest
import numpy as np
from hypertoroidal_wd_distribution import HypertoroidalWDDistribution
from toroidal_wd_distribution import ToroidalWDDistribution

class TestHypertoroidalWDDistribution(unittest.TestCase):

    def test_hypertoroidal_wd_distribution(self):
        d = np.array([[0.5, 3, 4, 6, 6],
                      [2, 2, 5, 3, 0],
                      [0.5, 0.2, 5.8, 4.3, 1.2]])
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
        np.testing.assert_almost_equal(m[0], np.sum(w * np.exp(1j * d[0, :])), decimal=10)
        np.testing.assert_almost_equal(m[1], np.sum(w * np.exp(1j * d[1, :])), decimal=10)

        # Test sampling
        n_samples = 5
        s = twd.sample(n_samples)
        self.assertEqual(s.shape, (d.shape[0], n_samples))
        np.testing.assert_array_almost_equal(s, np.mod(s, 2 * np.pi))

        # Test getMarginal
        for i in range(d.shape[0]):
            wd = twd.marginalize_to_1D(i)
            np.testing.assert_array_almost_equal(twd.w, wd.w)
            np.testing.assert_array_almost_equal(wd.d, np.reshape(twd.d[i, :], (1,-1)))

        # Test apply function
        same = twd.apply_function(lambda x: x)
        np.testing.assert_array_almost_equal(same.trigonometric_moment(1), twd.trigonometric_moment(1))
        shift_offset = np.array([1.4, -0.3, np.pi])
        shifted = twd.apply_function(lambda x: x + shift_offset)
        np.testing.assert_almost_equal(shifted.trigonometric_moment(1), twd.trigonometric_moment(1) * np.exp(1j * shift_offset), decimal=10)

    def test_shift(self):
        d = np.array([[0.5, 3, 4, 6, 6],
                      [2, 2, 5, 3, 0],
                      [0.5, 0.2, 5.8, 4.3, 1.2]])
        w = np.array([0.1, 0.1, 0.1, 0.1, 0.6])
        twd = HypertoroidalWDDistribution(d, w)
        s = np.array([1, -3, 6])
        twd_shifted = twd.shift(s)
        self.assertIsInstance(twd_shifted, HypertoroidalWDDistribution)
        np.testing.assert_array_almost_equal(twd.w, twd_shifted.w)
        np.testing.assert_array_almost_equal(twd.d, np.mod(twd_shifted.d - np.outer(s, np.ones_like(w)), 2 * np.pi), decimal=10)

    def test_to_toroidal_wd(self):
        np.random.seed(0)
        n = 20
        d = 2 * np.pi * np.random.rand(2, n)
        w = np.random.rand(1, n)
        w = w / np.sum(w)
        hwd = HypertoroidalWDDistribution(d, w)
        twd1 = ToroidalWDDistribution(d, w)
        twd2 = hwd.to_toroidal_wd()
        self.assertIsInstance(twd2, ToroidalWDDistribution)
        np.testing.assert_array_almost_equal(twd1.d, twd2.d, decimal=10)
        np.testing.assert_array_almost_equal(twd1.w, twd2.w, decimal=10)

    def test_marginalization(self):
        np.random.seed(0)
        n = 20
        d = 2 * np.pi * np.random.rand(2, n)
        w = np.random.rand(1, n)
        w = w / np.sum(w)
        hwd = HypertoroidalWDDistribution(d, w)
        wd1 = hwd.marginalize_to_1D(0)
        wd2 = hwd.marginalize_out(1)
        np.testing.assert_array_almost_equal(wd1.d, wd2.d)
        np.testing.assert_array_almost_equal(wd1.w, wd2.w)
    
if __name__ == '__main__':
    unittest.main()
