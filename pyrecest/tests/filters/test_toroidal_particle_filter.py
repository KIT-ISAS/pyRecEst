from math import pi
from pyrecest.backend import random
from pyrecest.backend import array
from pyrecest.backend import allclose
from pyrecest.backend import all
import unittest


from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)
from pyrecest.filters.toroidal_particle_filter import ToroidalParticleFilter


class ToroidalParticleFilterTest(unittest.TestCase):
    def test_toroidal_particle_filter(self):
        random.seed(0)
        C = array([[0.7, 0.4], [0.4, 0.6]])
        mu = array([1, 1]) + pi / 2
        hwnd = ToroidalWrappedNormalDistribution(mu, C)
        tpf = ToroidalParticleFilter(200)
        tpf.set_state(hwnd)
        forced_mean = array([1, 1])

        for _ in range(50):
            tpf.predict_identity(
                HypertoroidalWrappedNormalDistribution(array([0, 0]), C)
            )
            for _ in range(3):
                tpf.update_identity(
                    HypertoroidalWrappedNormalDistribution(array([0, 0]), 0.5 * C),
                    forced_mean,
                )

        self.assertTrue(allclose(tpf.get_point_estimate(), forced_mean, atol=0.1))


if __name__ == "__main__":
    unittest.main()