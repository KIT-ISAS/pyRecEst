import unittest

import numpy as np
from pyrecest.distributions import HypertoroidalWNDistribution
from pyrecest.filters import HypertoroidalParticleFilter

class HypertoroidalParticleFilterTest(unittest.TestCase):
    def setUp(self):
        self.seed = 0
        self.covariance_matrix = np.array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        self.mu = np.array([1, 1, 1]) + np.pi / 2
        self.hwnd = HypertoroidalWNDistribution(self.mu, self.covariance_matrix)
        self.hpf = HypertoroidalParticleFilter(500, 3)
        self.forced_mean = np.array([1, 2, 3])
        np.random.seed(self.seed)

    def test_set_state(self):
        self.hpf.set_state(self.hwnd)

    def test_predict_identity(self):
        self.hpf.predict_identity(HypertoroidalWNDistribution(np.zeros(3), 0.5 * self.covariance_matrix))
        self.assertEqual(self.hpf.get_point_estimate().shape, (3,))

    def test_update_identity(self):
        self.hpf.update_identity(HypertoroidalWNDistribution(np.zeros(3), 0.5 * self.covariance_matrix), self.forced_mean)
        self.assertEqual(self.hpf.get_point_estimate().shape, (3,))

    def test_predict_update_cycle_3D(self):
        self.hpf.set_state(self.hwnd)
        for _ in range(10):
            self.test_predict_identity()
            for _ in range(3):
                self.test_update_identity()
        np.testing.assert_allclose(self.hpf.get_point_estimate(), self.forced_mean, atol=0.1)

if __name__ == "__main__":
    unittest.main()
