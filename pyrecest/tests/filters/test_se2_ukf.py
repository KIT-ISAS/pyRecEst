"""Tests for the SE2UKF filter."""

import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,redefined-builtin
import pyrecest.backend
from pyrecest.backend import allclose, array, asarray, cos, eye, linalg, pi, sin, trace
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.se2_ukf import SE2UKF, _dual_quaternion_multiply


class TestDualQuaternionMultiply(unittest.TestCase):
    def test_identity_left(self):
        identity = array([1.0, 0.0, 0.0, 0.0])
        dq = array([0.7071, 0.7071, 0.5, -0.3])
        npt.assert_allclose(_dual_quaternion_multiply(identity, dq), dq, atol=1e-10)

    def test_identity_right(self):
        identity = array([1.0, 0.0, 0.0, 0.0])
        dq = array([0.7071, 0.7071, 0.5, -0.3])
        npt.assert_allclose(_dual_quaternion_multiply(dq, identity), dq, atol=1e-10)

    def test_non_commutativity(self):
        dq1 = array([cos(0.3), sin(0.3), 0.1, 0.2])
        dq2 = array([cos(0.5), sin(0.5), -0.1, 0.3])
        self.assertFalse(
            allclose(
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
        npt.assert_allclose(
            asarray(self.filter.filter_state.mu), [1.0, 0.0, 0.0, 0.0], atol=1e-15
        )

    def test_default_covariance_shape(self):
        self.assertEqual(self.filter.filter_state.C.shape, (4, 4))

    def test_set_state(self):
        mu = array([cos(0.3), sin(0.3), 1.0, -0.5])
        C = eye(4) * 0.01
        self.filter.filter_state = GaussianDistribution(mu, C)
        npt.assert_allclose(
            asarray(self.filter.filter_state.mu),
            asarray(mu),
            atol=1e-15,
        )

    def test_set_state_unnormalized_raises(self):
        mu = array([1.0, 1.0, 0.0, 0.0])
        C = eye(4) * 0.01
        g = GaussianDistribution(mu, C, check_validity=False)
        with self.assertRaises(AssertionError):
            self.filter.filter_state = g

    def test_get_point_estimate(self):
        mu = array([cos(0.3), sin(0.3), 1.0, -0.5])
        C = eye(4) * 0.01
        self.filter.filter_state = GaussianDistribution(mu, C)
        npt.assert_allclose(
            asarray(self.filter.get_point_estimate()),
            asarray(mu),
            atol=1e-15,
        )


class TestSE2UKFPredictIdentity(unittest.TestCase):
    def _make_noise(self, scale=0.01):
        return GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]), eye(4) * scale
        )

    def test_predict_preserves_type(self):
        f = SE2UKF()
        f.predict_identity(self._make_noise())
        self.assertIsInstance(f.filter_state, GaussianDistribution)

    def test_predict_increases_covariance(self):
        f = SE2UKF()
        C_before = trace(f.filter_state.C)
        f.predict_identity(self._make_noise(scale=0.1))
        C_after = trace(f.filter_state.C)
        self.assertGreater(C_after, C_before)

    def test_predict_mean_stays_normalised(self):
        f = SE2UKF()
        for _ in range(5):
            f.predict_identity(self._make_noise())
        npt.assert_allclose(linalg.norm(f.filter_state.mu[:2]), 1.0, atol=1e-10)

    def test_predict_covariance_symmetric(self):
        f = SE2UKF()
        f.predict_identity(self._make_noise(0.05))
        C = f.filter_state.C
        npt.assert_allclose(asarray(C), asarray(C).T, atol=1e-10)

    def test_predict_with_nonidentity_initial_state(self):
        f = SE2UKF()
        angle = pi / 6
        mu = array([cos(angle / 2), sin(angle / 2), 2.0, -1.0])
        f.filter_state = GaussianDistribution(mu, eye(4) * 0.01)
        f.predict_identity(self._make_noise())
        npt.assert_allclose(linalg.norm(f.filter_state.mu[:2]), 1.0, atol=1e-10)


class TestSE2UKFUpdateIdentity(unittest.TestCase):
    def _make_meas_noise(self, scale=0.01):
        return GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]), eye(4) * scale
        )

    def test_update_preserves_type(self):
        f = SE2UKF()
        f.update_identity(self._make_meas_noise(), array([1.0, 0.0, 0.0, 0.0]))
        self.assertIsInstance(f.filter_state, GaussianDistribution)

    def test_update_reduces_covariance(self):
        f = SE2UKF()
        C_before = trace(f.filter_state.C)
        f.update_identity(
            self._make_meas_noise(scale=0.1), array([1.0, 0.0, 0.0, 0.0])
        )
        C_after = trace(f.filter_state.C)
        self.assertLess(C_after, C_before)

    def test_update_mean_stays_normalised(self):
        f = SE2UKF()
        f.update_identity(
            self._make_meas_noise(),
            array([cos(0.5), sin(0.5), 0.3, -0.2]),
        )
        npt.assert_allclose(linalg.norm(f.filter_state.mu[:2]), 1.0, atol=1e-10)

    def test_update_covariance_symmetric(self):
        f = SE2UKF()
        f.update_identity(self._make_meas_noise(), array([1.0, 0.0, 0.0, 0.0]))
        C = f.filter_state.C
        npt.assert_allclose(asarray(C), asarray(C).T, atol=1e-10)

    def test_update_shifts_mean_toward_measurement(self):
        f = SE2UKF()
        angle = pi / 3
        z = array([cos(angle / 2), sin(angle / 2), 1.0, 0.5])
        mu_before = f.filter_state.mu
        f.update_identity(self._make_meas_noise(), z)
        mu_after = f.filter_state.mu
        self.assertLess(linalg.norm(mu_after - z), linalg.norm(mu_before - z))

    def test_update_antipodal_measurement(self):
        f1 = SE2UKF()
        f2 = SE2UKF()
        z = array([1.0, 0.0, 0.0, 0.0])
        f1.update_identity(self._make_meas_noise(), z)
        f2.update_identity(self._make_meas_noise(), -z)
        npt.assert_allclose(
            asarray(f1.filter_state.mu),
            asarray(f2.filter_state.mu),
            atol=1e-10,
        )

    def test_predict_then_update_cycle(self):
        f = SE2UKF()
        noise = self._make_meas_noise(scale=0.05)
        angle = pi / 4
        mu0 = array([cos(angle / 2), sin(angle / 2), 0.5, 0.2])
        f.filter_state = GaussianDistribution(mu0, eye(4) * 0.02)
        f.predict_identity(noise)
        z = array([cos(angle / 2), sin(angle / 2), 0.5, 0.2])
        f.update_identity(noise, z)
        npt.assert_allclose(linalg.norm(f.filter_state.mu[:2]), 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
