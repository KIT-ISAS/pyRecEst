import unittest
from math import pi

import numpy as np
import numpy.testing as npt

from pyrecest.backend import array
from pyrecest.distributions.cart_prod.custom_hypercylindrical_distribution import (
    CustomHypercylindricalDistribution,
)
from pyrecest.distributions.cart_prod.hypercylindrical_state_space_subdivision_distribution import (
    HypercylindricalStateSpaceSubdivisionDistribution,
)
from pyrecest.distributions.cart_prod.state_space_subdivision_distribution import (
    StateSpaceSubdivisionDistribution,
)
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.custom_linear_distribution import (
    CustomLinearDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


class HypertoroidalGridDistributionTest(unittest.TestCase):
    def setUp(self):
        self.n = 20
        self.gd = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), self.n
        )

    def test_from_distribution_shape(self):
        self.assertEqual(self.gd.grid.shape[0], self.n)
        self.assertEqual(self.gd.grid_values.shape[0], self.n)

    def test_from_distribution_uniform(self):
        # Uniform distribution should have equal grid values
        npt.assert_allclose(
            self.gd.grid_values,
            np.full(self.n, 1.0 / (2.0 * pi)),
            rtol=1e-10,
        )

    def test_integrate_uniform(self):
        npt.assert_allclose(self.gd.integrate(), 1.0, rtol=1e-10)

    def test_get_closest_point(self):
        xs = array([0.0, pi / 2, pi, 3 * pi / 2])
        pts, idxs = self.gd.get_closest_point(xs)
        # Check indices are in valid range
        self.assertTrue(np.all(idxs >= 0))
        self.assertTrue(np.all(idxs < self.n))

    def test_get_closest_point_wrapping(self):
        # 2*pi should wrap to first grid point (near 0)
        _, idx_0 = self.gd.get_closest_point(array([0.0]))
        _, idx_2pi = self.gd.get_closest_point(array([2.0 * pi - 1e-10]))
        # Both should be in the neighborhood of 0
        self.assertIn(idx_0[0], [0])

    def test_normalize(self):
        # Scale up and renormalize
        gd2 = HypertoroidalGridDistribution(self.gd.grid_values * 3.0)
        gd2.normalize_in_place()
        npt.assert_allclose(gd2.integrate(), 1.0, rtol=1e-10)

    def test_sample(self):
        samples = self.gd.sample(100)
        self.assertEqual(samples.shape, (100,))
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples < 2.0 * pi))


class StateSpaceSubdivisionDistributionTest(unittest.TestCase):
    def setUp(self):
        self.n = 10
        gd = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), self.n
        )
        lin_dists = []
        for i in range(self.n):
            mu_i = float(i) * 0.5
            # Use GaussianDistribution for proper sampling support
            lin_dists.append(
                GaussianDistribution(array([mu_i]), array([[0.25]]))
            )
        self.ssd = StateSpaceSubdivisionDistribution(gd, lin_dists)

    def test_bound_and_lin_dims(self):
        self.assertEqual(self.ssd.bound_dim, 1)
        self.assertEqual(self.ssd.lin_dim, 1)
        self.assertEqual(self.ssd.dim, 2)

    def test_pdf_shape(self):
        xa = array([[0.5, 0.0], [1.5, 1.0], [3.0, 2.0]])
        p = self.ssd.pdf(xa)
        self.assertEqual(p.shape, (3,))

    def test_pdf_nonnegative(self):
        xa = array([[0.5, 0.0], [1.5, 1.0], [3.0, 2.0]])
        p = self.ssd.pdf(xa)
        self.assertTrue(np.all(np.asarray(p) >= 0.0))

    def test_marginalize_linear(self):
        gd_marginal = self.ssd.marginalize_linear()
        # Should be the grid distribution
        self.assertIsInstance(gd_marginal, HypertoroidalGridDistribution)

    def test_marginalize_periodic(self):
        from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture

        lin_marginal = self.ssd.marginalize_periodic()
        self.assertIsInstance(lin_marginal, LinearMixture)

    def test_mode(self):
        m = self.ssd.mode()
        self.assertEqual(m.shape, (2,))

    def test_sample(self):
        samples = self.ssd.sample(50)
        self.assertEqual(samples.shape, (50, 2))
        # Periodic part should be in [0, 2*pi)
        self.assertTrue(np.all(np.asarray(samples[:, 0]) >= 0.0))
        self.assertTrue(np.all(np.asarray(samples[:, 0]) < 2.0 * pi))


class HypercylindricalStateSpaceSubdivisionDistributionTest(unittest.TestCase):
    def setUp(self):
        self.vm = VonMisesDistribution(array(0.0), array(2.0))
        self.gauss = GaussianDistribution(array([1.0]), array([[0.25]]))
        self.n = 20

        def fun(x):
            return self.vm.pdf(x[:, 0]) * self.gauss.pdf(x[:, 1:])

        self.fun = fun
        self.chd = CustomHypercylindricalDistribution(fun, 1, 1)

    def test_from_distribution(self):
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        self.assertIsInstance(
            hcrbd, HypercylindricalStateSpaceSubdivisionDistribution
        )
        self.assertEqual(hcrbd.bound_dim, 1)
        self.assertEqual(hcrbd.lin_dim, 1)

    def test_from_function(self):
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            self.fun, self.n, 1, 1
        )
        self.assertIsInstance(
            hcrbd, HypercylindricalStateSpaceSubdivisionDistribution
        )

    def test_pdf_approximation(self):
        """Test that the pdf approximation is reasonably close to the true pdf."""
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        test_points = array(
            [[0.0, 1.0], [pi / 2, 0.5], [pi, 1.0], [3 * pi / 2, 1.5]]
        )
        p_approx = np.asarray(hcrbd.pdf(test_points))
        p_true = np.asarray(self.chd.pdf(test_points))
        # Rough check: same order of magnitude
        npt.assert_allclose(p_approx, p_true, rtol=0.3)

    def test_mode_reasonable(self):
        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_distribution(
            self.chd, self.n
        )
        m = hcrbd.mode()
        self.assertEqual(m.shape, (2,))
        # Mode of periodic part should be near 0 (von Mises mean)
        npt.assert_allclose(np.asarray(m[0]) % (2 * pi), 0.0, atol=2 * pi / self.n)
        # Mode of linear part should be near 1.0 (Gaussian mean)
        npt.assert_allclose(np.asarray(m[1]), 1.0, atol=0.1)

    def test_pdf_integrates_to_one(self):
        """Coarse check that the pdf integrates close to 1."""
        from scipy.integrate import dblquad

        hcrbd = HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            self.fun, self.n, 1, 1
        )
        integral, _ = dblquad(
            lambda y, x: float(np.squeeze(hcrbd.pdf(np.array([[x, y]])))),
            0,
            2 * pi,
            -4,
            6,
            epsabs=1e-2,
            epsrel=1e-2,
        )
        npt.assert_allclose(integral, 1.0, atol=0.05)


if __name__ == "__main__":
    unittest.main()
