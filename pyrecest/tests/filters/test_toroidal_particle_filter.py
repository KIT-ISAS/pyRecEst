import unittest

import numpy as np
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)
from pyrecest.filters.toroidal_particle_filter import ToroidalParticleFilter


class ToroidalParticleFilterTest(unittest.TestCase):
    def test_toroidal_particle_filter(self):
        np.random.seed(0)
        C = np.array([[0.7, 0.4], [0.4, 0.6]])
        mu = np.array([1, 1]) + np.pi / 2
        hwnd = ToroidalWrappedNormalDistribution(mu, C)
        tpf = ToroidalParticleFilter(200)
        tpf.set_state(hwnd)
        forced_mean = np.array([1, 1])

        for _ in range(50):
            tpf.predict_identity(
                HypertoroidalWrappedNormalDistribution(np.array([0, 0]), C)
            )
            for _ in range(3):
                tpf.update_identity(
                    HypertoroidalWrappedNormalDistribution(np.array([0, 0]), 0.5 * C),
                    forced_mean,
                )

        self.assertTrue(np.allclose(tpf.get_point_estimate(), forced_mean, atol=0.1))


if __name__ == "__main__":
    unittest.main()
