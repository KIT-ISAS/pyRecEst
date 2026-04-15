"""Tests for the SE2UKF filter."""

import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.se2_ukf import SE2UKF, _dual_quaternion_multiply


class TestDualQuaternionMultiply(unittest.TestCase):
    def test_identity_left(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        dq = np.array([0.7071, 0.7071, 0.5, -0.3])
        npt.assert_allclose(_dual_quaternion_multiply(identity, dq), dq, atol=1e-10)

    def test_identity_right(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        dq = np.array([0.7071, 0.7071, 0.5, -0.3])
        npt.assert_allclose(_dual_quaternion_multiply(dq, identity), dq, atol=1e-10)

    def test_non_commutativity(self):
        dq1 = np.array([np.cos(0.3), np.sin(0.3), 0.1, 0.2])
        dq2 = np.array([np.cos(0.5), np.sin(0.5), -0.1, 0.3])
        self.assertFalse(
            np.allclose(
                _dual_quaternion_multiply(dq1, dq2),
                _dual_quaternion_multiply(dq2, dq1),
            )
        )


class TestSE2UKFInitialization(unittest.TestCase):
    def setUp(self):
        self.filter = SE2UKF()

    def test_default_state_is_gaussian(self):
        self.assertIsInstance(self.filter.filter_state, GaussianDistribution)

    def test_default_mean_is_identity(self):
        mu = np.asarray(self.filter.filter_state.mu, dtype=float)
        npt.assert_allclose(mu, [1.0, 0.0, 0.0, 0.0], atol=1e-15)

    def test_default_covariance_shape(self):
        self.assertEqual(self.filter.filter_state.C.shape, (4, 4))

    def test_set_state(self):
        mu = array([np.cos(0.3), np.sin(0.3), 1.0, -0.5])
        C = array(np.eye(4) * 0.01)
        self.filter.filter_state = GaussianDistribution(mu, C)
        npt.assert_allclose(
            np.asarray(self.filter.filter_state.mu, dtype=float),
            np.asarray(mu, dtype=float),
            atol=1e-15,
        )

    def test_set_state_unnormalized_raises(self):
        mu = array([1.0, 1.0, 0.0, 0.0])
        C = array(np.eye(4) * 0.01)
        g = GaussianDistribution(mu, C, check_validity=False)
        with self.assertRaises(AssertionError):
            self.filter.filter_state = g

    def test_get_point_estimate(self):
        mu = array([np.cos(0.3), np.sin(0.3), 1.0, -0.5])
        C = array(np.eye(4) * 0.01)
        self.filter.filter_state = GaussianDistribution(mu, C)
        npt.assert_allclose(
            np.asarray(self.filter.get_point_estimate(), dtype=float),
            np.asarray(mu, dtype=float),
            atol=1e-15,
        )


class TestSE2UKFPredictIdentity(unittest.TestCase):
    def _make_noise(self, scale=0.01):
        return GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]), array(np.eye(4) * scale)
        )

    def test_predict_preserves_type(self):
        f = SE2UKF()
        f.predict_identity(self._make_noise())
        self.assertIsInstance(f.filter_state, GaussianDistribution)

    def test_predict_increases_covariance(self):
        f = SE2UKF()
        C_before = np.trace(np.asarray(f.filter_state.C, dtype=float))
        f.predict_identity(self._make_noise(scale=0.1))
        C_after = np.trace(np.asarray(f.filter_state.C, dtype=float))
        self.assertGreater(C_after, C_before)

    def test_predict_mean_stays_normalised(self):
        f = SE2UKF()
        for _ in range(5):
            f.predict_identity(self._make_noise())
        mu = np.asarray(f.filter_state.mu, dtype=float)
        npt.assert_allclose(np.linalg.norm(mu[:2]), 1.0, atol=1e-10)

    def test_predict_covariance_symmetric(self):
        f = SE2UKF()
        f.predict_identity(self._make_noise(0.05))
        C = np.asarray(f.filter_state.C, dtype=float)
        npt.assert_allclose(C, C.T, atol=1e-10)

    def test_predict_with_nonidentity_initial_state(self):
        f = SE2UKF()
        angle = np.pi / 6
        mu = array([np.cos(angle / 2), np.sin(angle / 2), 2.0, -1.0])
        f.filter_state = GaussianDistribution(mu, array(np.eye(4) * 0.01))
        f.predict_identity(self._make_noise())
        mu_after = np.asarray(f.filter_state.mu, dtype=float)
        npt.assert_allclose(np.linalg.norm(mu_after[:2]), 1.0, atol=1e-10)


class TestSE2UKFUpdateIdentity(unittest.TestCase):
    def _make_meas_noise(self, scale=0.01):
        return GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]), array(np.eye(4) * scale)
        )

    def test_update_preserves_type(self):
        f = SE2UKF()
        f.update_identity(self._make_meas_noise(), np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertIsInstance(f.filter_state, GaussianDistribution)

    def test_update_reduces_covariance(self):
        f = SE2UKF()
        C_before = np.trace(np.asarray(f.filter_state.C, dtype=float))
        f.update_identity(
            self._make_meas_noise(scale=0.1), np.array([1.0, 0.0, 0.0, 0.0])
        )
        C_after = np.trace(np.asarray(f.filter_state.C, dtype=float))
        self.assertLess(C_after, C_before)

    def test_update_mean_stays_normalised(self):
        f = SE2UKF()
        f.update_identity(
            self._make_meas_noise(),
            np.array([np.cos(0.5), np.sin(0.5), 0.3, -0.2]),
        )
        mu = np.asarray(f.filter_state.mu, dtype=float)
        npt.assert_allclose(np.linalg.norm(mu[:2]), 1.0, atol=1e-10)

    def test_update_covariance_symmetric(self):
        f = SE2UKF()
        f.update_identity(self._make_meas_noise(), np.array([1.0, 0.0, 0.0, 0.0]))
        C = np.asarray(f.filter_state.C, dtype=float)
        npt.assert_allclose(C, C.T, atol=1e-10)

    def test_update_shifts_mean_toward_measurement(self):
        f = SE2UKF()
        angle = np.pi / 3
        z = np.array([np.cos(angle / 2), np.sin(angle / 2), 1.0, 0.5])
        mu_before = np.asarray(f.filter_state.mu, dtype=float)
        f.update_identity(self._make_meas_noise(), z)
        mu_after = np.asarray(f.filter_state.mu, dtype=float)
        self.assertLess(np.linalg.norm(mu_after - z), np.linalg.norm(mu_before - z))

    def test_update_antipodal_measurement(self):
        f1 = SE2UKF()
        f2 = SE2UKF()
        z = np.array([1.0, 0.0, 0.0, 0.0])
        f1.update_identity(self._make_meas_noise(), z)
        f2.update_identity(self._make_meas_noise(), -z)
        npt.assert_allclose(
            np.asarray(f1.filter_state.mu, dtype=float),
            np.asarray(f2.filter_state.mu, dtype=float),
            atol=1e-10,
        )

    def test_predict_then_update_cycle(self):
        f = SE2UKF()
        noise = self._make_meas_noise(scale=0.05)
        angle = np.pi / 4
        mu0 = array([np.cos(angle / 2), np.sin(angle / 2), 0.5, 0.2])
        f.filter_state = GaussianDistribution(mu0, array(np.eye(4) * 0.02))
        f.predict_identity(noise)
        z = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.5, 0.2])
        f.update_identity(noise, z)
        mu = np.asarray(f.filter_state.mu, dtype=float)
        npt.assert_allclose(np.linalg.norm(mu[:2]), 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
