import numpy as np
import unittest

from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


class EuclideanParticleFilterTest(unittest.TestCase):

    def test_predict_update_cycle_3d(self):
        np.random.seed(0)
        C = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = np.array([5, 6, 7])
        prior = GaussianDistribution(mu, C)
        pf = EuclideanParticleFilter(n_particles=500, dim=3)
        pf.set_state(prior)
        forced_mean = np.array([1, 2, 3])
        for _ in range(50):
            pf.predict_identity(GaussianDistribution(np.zeros(3), C))
            self.assertEqual(pf.get_point_estimate().shape, (3,))
            for _ in range(3):
                pf.update_identity(GaussianDistribution(np.zeros(3), 0.5 * C), forced_mean)

        self.assertEqual(pf.get_point_estimate().shape, (3,))
        np.testing.assert_almost_equal(pf.get_point_estimate(), forced_mean, decimal=1)

        n = 5
        samples = np.random.rand(n, 3)
        weights = np.ones((n)) / n
        f = lambda x, w: x + w
        pf.set_state(prior)
        pf.predict_nonlinear_nonadditive(f, samples, weights)
        est = pf.get_point_estimate()
        self.assertEqual(pf.get_point_estimate().shape, (3,))
        np.testing.assert_allclose(est, prior.mu + np.mean(samples, axis=0), atol=0.1)

    def test_predict_update_cycle_3d_forced_particle_pos_no_pred(self):
        np.random.seed(0)
        C = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = np.ones(3) + np.pi / 2
        prior = GaussianDistribution(mu, C)
        pf = EuclideanParticleFilter(500, 3)
        pf.set_state(prior)
        forced_mean = np.array([3, 2, 1])
        force_first_particle_pos = np.array([3.1, 2, 1])
        pf.filter_state.d[0, :] = force_first_particle_pos
        for _ in range(50):
            self.assertEqual(pf.get_point_estimate().shape, (3,))
            for _ in range(3):
                pf.update_identity(GaussianDistribution(np.zeros(3), 0.5 * C), forced_mean)

        self.assertEqual(pf.get_point_estimate().shape, (3,))
        np.testing.assert_allclose(pf.get_point_estimate(), force_first_particle_pos, atol=1e-10)

if __name__ == "__main__":
    unittest.main()
