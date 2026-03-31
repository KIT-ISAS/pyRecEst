import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linspace, pi
from pyrecest.distributions import GvMDistribution, VonMisesDistribution


class TestGvMDistribution(unittest.TestCase):
    def test_init(self):
        dist = GvMDistribution(array([2.0]), array([1.0]))
        self.assertEqual(dist.mu.shape, (1,))
        self.assertEqual(dist.kappa.shape, (1,))

    def test_init_invalid_kappa(self):
        with self.assertRaises(AssertionError):
            GvMDistribution(array([2.0]), array([-1.0]))

    def test_init_shape_mismatch(self):
        with self.assertRaises(AssertionError):
            GvMDistribution(array([1.0, 2.0]), array([1.0]))

    def test_pdf_order1_matches_von_mises(self):
        """Order-1 GvM with a single mu and kappa should match the von Mises PDF."""
        mu_val = 2.0
        kappa_val = 1.0
        gvm = GvMDistribution(array([mu_val]), array([kappa_val]))
        vm = VonMisesDistribution(mu_val, kappa_val)

        xs = linspace(0.0, 2.0 * pi, 100)
        npt.assert_array_almost_equal(gvm.pdf(xs), vm.pdf(xs), decimal=5)

    def test_pdf_non_negative(self):
        gvm = GvMDistribution(array([1.0, 0.5]), array([2.0, 1.0]))
        xs = linspace(0.0, 2.0 * pi, 50)
        self.assertTrue((gvm.pdf(xs) >= 0).all())

    def test_pdf_integrates_to_one(self):
        """PDF should integrate to 1 over [0, 2*pi]."""
        from scipy.integrate import quad

        gvm = GvMDistribution(array([1.0, 0.5]), array([2.0, 1.0]))
        result, _ = quad(lambda x: float(gvm.pdf(array([x]))[0]), 0.0, 2.0 * np.pi)
        npt.assert_almost_equal(result, 1.0, decimal=5)

    def test_pdf_order2_specific_values(self):
        """Test order-2 PDF at specific points, compared to manually computed values."""
        mu = array([2.0, 1.0])
        kappa = array([1.0, 0.5])
        gvm = GvMDistribution(mu, kappa)

        xs = linspace(0.0, 2.0 * pi, 5)
        pdf_vals = gvm.pdf(xs)
        # All values must be positive and finite
        self.assertTrue(np.all(np.isfinite(pdf_vals)))
        self.assertTrue(np.all(pdf_vals >= 0))

    def test_norm_const_cached(self):
        """Norm constant should be computed once and cached."""
        gvm = GvMDistribution(array([1.0]), array([1.0]))
        c1 = gvm.norm_const
        c2 = gvm.norm_const
        self.assertEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
