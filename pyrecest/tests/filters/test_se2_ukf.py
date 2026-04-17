import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, eye, pi, sin
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.se2_ukf import SE2UKF, _dual_quaternion_multiply


def _make_identity_state(scale=0.1):
    """Return a GaussianDistribution centred at the SE(2) identity."""
    mu = array([1.0, 0.0, 0.0, 0.0])
    C = eye(4) * scale
    return GaussianDistribution(mu, C)


def _make_noise(scale=0.01):
    """Return a small-noise GaussianDistribution at the SE(2) identity."""
    mu = array([1.0, 0.0, 0.0, 0.0])
    C = eye(4) * scale
    return GaussianDistribution(mu, C)


class TestDualQuaternionMultiply(unittest.TestCase):
    def test_identity_left(self):
        """Multiplying by the identity from the left leaves dq unchanged."""
        identity = array([1.0, 0.0, 0.0, 0.0])
        dq = array([0.7071, 0.7071, 0.3, -0.1])
        result = _dual_quaternion_multiply(identity, dq)
        npt.assert_allclose(np.array(result), np.array(dq), atol=1e-10)

    def test_identity_right(self):
        """Multiplying by the identity from the right leaves dq unchanged."""
        identity = array([1.0, 0.0, 0.0, 0.0])
        dq = array([0.7071, 0.7071, 0.3, -0.1])
        result = _dual_quaternion_multiply(dq, identity)
        npt.assert_allclose(np.array(result), np.array(dq), atol=1e-10)

    def test_rotation_composition(self):
        """Composing two 90-degree rotations gives a 180-degree rotation."""
        theta = pi / 2
        dq_90 = array([cos(theta / 2), sin(theta / 2), 0.0, 0.0])
        result = _dual_quaternion_multiply(dq_90, dq_90)
        # cos(pi/2) == 0, sin(pi/2) == 1 (exact mathematical values)
        expected = array([0.0, 1.0, 0.0, 0.0])
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-6)

    def test_result_shape(self):
        dq1 = array([1.0, 0.0, 0.0, 0.0])
        dq2 = array([0.0, 1.0, 0.0, 0.0])
        result = _dual_quaternion_multiply(dq1, dq2)
        self.assertEqual(result.shape, (4,))


class TestSE2UKF(unittest.TestCase):
    def setUp(self):
        self.filter = SE2UKF()

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_initial_state_is_gaussian(self):
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_set_state(self):
        state = _make_identity_state()
        self.filter.filter_state = state
        fs = self.filter.filter_state
        self.assertIsInstance(fs, GaussianDistribution)
        npt.assert_allclose(np.array(fs.mu), np.array(state.mu), atol=1e-10)
        npt.assert_allclose(np.array(fs.C), np.array(state.C), atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_set_state_unnormalised_raises(self):
        mu = array([1.0, 1.0, 0.0, 0.0])  # not unit-norm in first 2 entries
        C = eye(4) * 0.1
        with self.assertRaises(AssertionError):
            self.filter.filter_state = GaussianDistribution(mu, C)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_returns_gaussian(self):
        self.filter.filter_state = _make_identity_state()
        noise = _make_noise()
        self.filter.predict_identity(noise)
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_mean_normalised(self):
        """After prediction the rotation part of the mean must stay normalised."""
        self.filter.filter_state = _make_identity_state()
        self.filter.predict_identity(_make_noise())
        mu = np.array(self.filter.filter_state.mu)
        npt.assert_allclose(np.linalg.norm(mu[0:2]), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_covariance_shape(self):
        self.filter.filter_state = _make_identity_state()
        self.filter.predict_identity(_make_noise())
        C = np.array(self.filter.filter_state.C)
        self.assertEqual(C.shape, (4, 4))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_covariance_symmetric(self):
        self.filter.filter_state = _make_identity_state()
        self.filter.predict_identity(_make_noise())
        C = np.array(self.filter.filter_state.C)
        npt.assert_allclose(C, C.T, atol=1e-12)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_identity_covariance_increases(self):
        """Adding process noise should make the covariance (trace) larger."""
        self.filter.filter_state = _make_identity_state(scale=0.05)
        C_before = np.trace(np.array(self.filter.filter_state.C))
        self.filter.predict_identity(_make_noise(scale=0.05))
        C_after = np.trace(np.array(self.filter.filter_state.C))
        self.assertGreater(C_after, C_before)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_update_identity_returns_gaussian(self):
        self.filter.filter_state = _make_identity_state()
        z = array([1.0, 0.0, 0.0, 0.0])
        self.filter.update_identity(_make_noise(), z)
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_update_identity_mean_normalised(self):
        """After update the rotation part of the mean must stay normalised."""
        self.filter.filter_state = _make_identity_state()
        z = array([1.0, 0.0, 0.5, -0.3])
        z_np = np.array(z)
        z_np[0:2] /= np.linalg.norm(z_np[0:2])
        self.filter.update_identity(_make_noise(), z_np)
        mu = np.array(self.filter.filter_state.mu)
        npt.assert_allclose(np.linalg.norm(mu[0:2]), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_update_identity_covariance_decreases(self):
        """An informative measurement should reduce uncertainty."""
        self.filter.filter_state = _make_identity_state(scale=0.5)
        C_before = np.trace(np.array(self.filter.filter_state.C))
        z = array([1.0, 0.0, 0.0, 0.0])
        self.filter.update_identity(_make_noise(scale=0.01), z)
        C_after = np.trace(np.array(self.filter.filter_state.C))
        self.assertLess(C_after, C_before)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_predict_then_update_cycle(self):
        """A full predict+update cycle should leave the state as a GaussianDistribution
        with a normalised rotation part in the mean."""
        self.filter.filter_state = _make_identity_state()
        self.filter.predict_identity(_make_noise())
        z = array([1.0, 0.0, 0.1, -0.05])
        z_np = np.array(z)
        z_np[0:2] /= np.linalg.norm(z_np[0:2])
        self.filter.update_identity(_make_noise(), z_np)
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)
        mu = np.array(self.filter.filter_state.mu)
        npt.assert_allclose(np.linalg.norm(mu[0:2]), 1.0, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on JAX backend",
    )
    def test_get_point_estimate(self):
        self.filter.filter_state = _make_identity_state()
        est = self.filter.get_point_estimate()
        npt.assert_allclose(np.array(est), [1.0, 0.0, 0.0, 0.0], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
