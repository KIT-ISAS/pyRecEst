import unittest
from pyrecest.distributions import VonMisesFisherDistribution, HypersphericalDiracDistribution
from pyrecest.filters.hyperspherical_particle_filter import HypersphericalParticleFilter
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, allclose, ones, linalg, random, exp
import numpy.testing as npt
# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
class HypersphericalParticleFilterTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test3D(self):
        random.seed(1)

        n_samples = 20000
        vmf_init_mean = array([1.0, 0.0, 0.0])
        vmf_init = VonMisesFisherDistribution(vmf_init_mean, 10)
        vmf_sys = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 10)
        vmf_meas = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 3)

        # Initialize filter
        hpf = HypersphericalParticleFilter(n_samples, 3)
        hpf.filter_state = HypersphericalDiracDistribution(vmf_init.sample(n_samples))
        npt.assert_allclose(hpf.get_point_estimate(), vmf_init_mean, atol=1e-2)

        # Prediction step
        hpf.predict_identity(vmf_sys)
        npt.assert_allclose(hpf.get_point_estimate(), vmf_init_mean, atol=1e-2)

        # Update steps
        enforced_mean = array([0.0, 0.0, 1.0])
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.predict_identity(vmf_sys)
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.update_identity(vmf_meas, enforced_mean)
        hpf.update_identity(vmf_meas, enforced_mean)

        hpf_est_mean = hpf.get_point_estimate()
        self.assertEqual(hpf_est_mean.shape, (3,))
        npt.assert_allclose(hpf_est_mean, enforced_mean, atol=0.25)

        # Reset state
        hpf.filter_state = HypersphericalDiracDistribution(vmf_init.sample(n_samples))

        # Nonlinear update with a function that returns ones
        def f1(_, x):
            return ones(x.shape[0])

        z = array(3.0)
        est = hpf.get_point_estimate()
        npt.assert_allclose(linalg.norm(est), 1, atol=1e-10)
        # Since the likelihood is constant, the estimate should not change.
        hpf.update_nonlinear(f1, z)
        npt.assert_allclose(est, hpf.get_point_estimate(), atol=1e-2)

        z = array([0.0, 1.0, 0.0])
        def f2(z, x):
            return exp(z @ x.T)
        
        hpf.update_nonlinear(f2, z)
        est = hpf.get_point_estimate()
        npt.assert_allclose(linalg.norm(est), 1, atol=1e-10)
        self.assertFalse(allclose(est, vmf_init_mean, atol=1e-2))
