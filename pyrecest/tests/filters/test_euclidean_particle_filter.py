import copy
import unittest

import numpy as np
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


class EuclideanParticleFilterTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.C_prior = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        self.mu = np.array([5, 6, 7])
        self.prior = GaussianDistribution(self.mu, self.C_prior)
        self.sys_noise_default = GaussianDistribution(
            np.zeros_like(self.mu), 0.5 * self.C_prior
        )
        self.pf = EuclideanParticleFilter(n_particles=500, dim=3)
        self.forced_mean = np.array([1, 2, 3])
        self.pf.filter_state = self.prior

    def test_predict_update_cycle_3d(self):
        for _ in range(50):
            self.pf.predict_identity(GaussianDistribution(np.zeros(3), self.C_prior))
            # jscpd:ignore-start
            self.assertEqual(self.pf.get_point_estimate().shape, (3,))
            for _ in range(3):
                self.pf.update_identity(self.sys_noise_default, self.forced_mean)
            # jscpd:ignore-end

        self.assertEqual(self.pf.get_point_estimate().shape, (3,))
        np.testing.assert_almost_equal(
            self.pf.get_point_estimate(), self.forced_mean, decimal=1
        )

    def test_predict_nonlinear_nonadditive(self):
        n = 5
        samples = np.random.rand(n, 3)
        weights = np.ones((n)) / n

        def f(x, w):
            return x + w
        self.pf.predict_nonlinear_nonadditive(f, samples, weights)
        est = self.pf.get_point_estimate()
        self.assertEqual(self.pf.get_point_estimate().shape, (3,))
        np.testing.assert_allclose(
            est, self.prior.mu + np.mean(samples, axis=0), atol=0.1
        )

    def test_predict_update_cycle_3d_forced_particle_pos_no_pred(self):
        self.pf.filter_state = self.prior.set_mean(np.ones(3) + np.pi / 2)

        force_first_particle_pos = np.array([1.1, 2, 3])
        self.pf.filter_state.d[0, :] = force_first_particle_pos
        for _ in range(50):
            # jscpd:ignore-start
            self.assertEqual(self.pf.get_point_estimate().shape, (3,))
            for _ in range(3):
                self.pf.update_identity(self.sys_noise_default, self.forced_mean)
            # jscpd:ignore-end

        self.assertEqual(self.pf.get_point_estimate().shape, (3,))
        np.testing.assert_allclose(
            self.pf.get_point_estimate(), force_first_particle_pos, atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()
