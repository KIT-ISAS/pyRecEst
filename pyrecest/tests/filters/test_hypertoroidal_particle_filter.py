""" Particle filter for hypertoroidal domains """
import unittest

import numpy as np
from pyrecest.distributions import HypertoroidalWNDistribution
from pyrecest.filters import HypertoroidalParticleFilter


class HypertoroidalParticleFilterTest(unittest.TestCase):
    """Particle filter for hypertoroidal domains"""

    def test_predict_update_cycle_3D(self):
        np.random.seed(0)
        C = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = np.array([1, 1, 1]) + np.pi / 2
        hwnd = HypertoroidalWNDistribution(mu, C)
        hpf = HypertoroidalParticleFilter(500, 3)
        hpf.set_state(hwnd)
        forced_mean = np.array([1, 2, 3])

        for _ in range(10):
            hpf.predict_identity(HypertoroidalWNDistribution(np.zeros(3), 0.5 * C))
            self.assertEqual(hpf.get_point_estimate().shape, (3,))
            hpf.update_identity(
                HypertoroidalWNDistribution(np.zeros(3), 0.5 * C), forced_mean
            )
            hpf.update_identity(
                HypertoroidalWNDistribution(np.zeros(3), 0.5 * C), forced_mean
            )
            hpf.update_identity(
                HypertoroidalWNDistribution(np.zeros(3), 0.5 * C), forced_mean
            )

        self.assertEqual(hpf.get_point_estimate().shape, (3,))
        np.testing.assert_allclose(hpf.get_point_estimate(), forced_mean, atol=0.1)

    """
    def test_predict_update_cycle_3D_forced_particle_pos_no_pred(self):
        np.random.seed(0)
        C = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        mu = np.array([1, 1, 1]) + np.pi / 2
        hwnd = HypertoroidalWNDistribution(mu, C)
        hpf = HypertoroidalParticleFilter(500, 3)
        hpf.set_state(hwnd)
        forced_mean = np.array([1, 2, 3])
        force_first_particle_pos = np.array([1.1, 2, 3])
        hpf.dist.d[:, 0] = force_first_particle_pos

        for _ in range(50):
            self.assertEqual(hpf.get_estimate_mean().shape, (3, 1))
            hpf.update_identity(HypertoroidalWNDistribution(np.zeros(3), 0.5 * C), forced_mean)
            hpf.update_identity(HypertoroidalWNDistribution(np.zeros(3), 0.5 * C), forced_mean)
            hpf.update_identity(HypertoroidalWNDistribution(np.zeros(3), 0.5 * C), forced_mean)

        self.assertEqual(hpf.get_estimate_mean().shape, (3, 1))
        np.testing.assert_allclose(hpf.get_estimate_mean(), force_first_particle_pos[:, np.newaxis], atol=1e-14)
        """


if __name__ == "__main__":
    unittest.main()
