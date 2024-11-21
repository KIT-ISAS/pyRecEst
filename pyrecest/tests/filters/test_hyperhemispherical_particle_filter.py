import numpy as np
import unittest
from pyrecest.filters.hyperhemispherical_particle_filter import HyperhemisphericalParticleFilter
from pyrecest.distributions import HyperhemisphericalWatsonDistribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import HyperhemisphericalDiracDistribution

class HyperhemisphericalParticleFilterTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        self.n_particles = 2000
        self.watson_hemi_init = HyperhemisphericalWatsonDistribution(np.array([1, 0, 0]), 10)
        self.watson_hemi_sys = HyperhemisphericalWatsonDistribution(np.array([0, 0, 1]), 10)
        self.watson_hemi_meas = HyperhemisphericalWatsonDistribution(np.array([0, 0, 1]), 3)
        
    def test_initialization(self):
        hpf = HyperhemisphericalParticleFilter(self.n_particles, 3)
        self.assertEqual(np.shape(hpf.filter_state.w), (self.n_particles,))
        self.assertEqual(np.shape(hpf.filter_state.d), (self.n_particles, 4))

    def test_set_state(self):
        self.hpf = HyperhemisphericalParticleFilter(self.n_particles, 2)
        self.hpf.set_state(self.watson_hemi_init)
        point_est = self.hpf.get_point_estimate()
        watson_mode = self.watson_hemi_init.mode()
        is_close_to_mode = np.linalg.norm(point_est - watson_mode) < 1e-1
        is_close_to_neg_mode = np.linalg.norm(point_est - -watson_mode) < 1e-1
        self.assertTrue(is_close_to_mode or is_close_to_neg_mode)
        self.assertIsInstance(self.hpf.filter_state, HyperhemisphericalDiracDistribution)
        
        
if __name__ == "__main__":
    unittest.main()
