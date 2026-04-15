"""Tests for UKFOnManifolds."""
import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.filters.ukf_on_manifolds import UKFOnManifolds


# ---------------------------------------------------------------------------
# Helpers: Euclidean manifold (trivial phi / phi_inv, for numerical checks)
# ---------------------------------------------------------------------------

def _phi_euclidean(state, xi):
    return state + xi


def _phi_inv_euclidean(state_ref, state):
    return state - state_ref


# ---------------------------------------------------------------------------
# Helpers: SO(2) minimal example
# ---------------------------------------------------------------------------

def _phi_so2(state, xi):
    """Retraction on SO(2): state is a scalar angle, xi is a scalar tangent."""
    return state + xi[0]


def _phi_inv_so2(state_ref, state):
    """Inverse retraction on SO(2)."""
    diff = state - state_ref
    # wrap to [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return np.array([diff])


class TestUKFOnManifoldsEuclidean(unittest.TestCase):
    """Verify the filter recovers Kalman-filter results on a Euclidean state space."""

    def _make_filter(self, d=1, alpha=1e-3):
        Q = np.eye(d) * 0.1
        R = np.eye(d) * 0.5
        state0 = np.zeros(d)
        P0 = np.eye(d)

        def f(s, omega, w, dt):  # pylint: disable=unused-argument
            return s + w

        def h(s):
            return s

        return UKFOnManifolds(
            f=f,
            h=h,
            phi=_phi_euclidean,
            phi_inv=_phi_inv_euclidean,
            Q=Q,
            R=R,
            alpha=alpha,
            state0=state0,
            P0=P0,
        )

    # ------------------------------------------------------------------
    def test_initialization_state(self):
        ukf = self._make_filter(d=2)
        state, P = ukf.filter_state
        npt.assert_array_equal(state, np.zeros(2))
        npt.assert_array_equal(P, np.eye(2))

    def test_get_point_estimate_initial(self):
        ukf = self._make_filter(d=2)
        npt.assert_array_equal(ukf.get_point_estimate(), np.zeros(2))

    def test_predict_identity_increases_covariance(self):
        """After a predict step (no control), P should increase by Q."""
        ukf = self._make_filter(d=2)
        _, P_before = ukf.filter_state
        ukf.predict(omega=None, dt=1.0)
        _, P_after = ukf.filter_state
        # P should increase (P_after >= P_before elementwise for diagonal)
        self.assertTrue(np.all(np.diag(P_after) >= np.diag(P_before)))

    def test_update_reduces_covariance(self):
        """After an update step, P should be smaller."""
        ukf = self._make_filter(d=2)
        ukf.predict(omega=None, dt=1.0)
        _, P_before_update = ukf.filter_state
        ukf.update(y=np.array([0.5, -0.5]))
        _, P_after_update = ukf.filter_state
        # Covariance should decrease after update
        self.assertTrue(np.all(np.diag(P_after_update) < np.diag(P_before_update)))

    def test_update_moves_state_toward_measurement(self):
        """The state estimate should move toward the measurement after update."""
        ukf = self._make_filter(d=1)
        ukf.predict(omega=None, dt=1.0)
        y = np.array([2.0])
        state_before, _ = ukf.filter_state
        ukf.update(y=y)
        state_after, _ = ukf.filter_state
        # state should have moved in the direction of the measurement
        self.assertGreater(float(state_after[0]), float(state_before[0]))
        self.assertLess(float(state_after[0]), float(y[0]))

    def test_filter_state_setter_getter_roundtrip(self):
        ukf = self._make_filter(d=2)
        new_state = np.array([1.0, 2.0])
        new_P = np.diag([3.0, 4.0])
        ukf.filter_state = (new_state, new_P)
        state_out, P_out = ukf.filter_state
        npt.assert_array_equal(state_out, new_state)
        npt.assert_array_equal(P_out, new_P)

    def test_alpha_scalar_vs_array_equivalent(self):
        """Passing a scalar alpha should give the same result as [alpha, alpha, alpha]."""
        Q = np.eye(2) * 0.1
        R = np.eye(2) * 0.5
        state0 = np.zeros(2)
        P0 = np.eye(2)
        measurement = np.array([1.0, -1.0])

        def f(s, omega, w, dt):  # pylint: disable=unused-argument
            return s + w

        def h(s):
            return s

        common = dict(f=f, h=h, phi=_phi_euclidean, phi_inv=_phi_inv_euclidean,
                      Q=Q, R=R, state0=state0, P0=P0)
        ukf_scalar = UKFOnManifolds(alpha=1e-3, **common)
        ukf_array = UKFOnManifolds(alpha=[1e-3, 1e-3, 1e-3], **common)

        for ukf in (ukf_scalar, ukf_array):
            ukf.predict(omega=None, dt=1.0)
            ukf.update(y=measurement)

        state_s, P_s = ukf_scalar.filter_state
        state_a, P_a = ukf_array.filter_state
        npt.assert_allclose(state_s, state_a)
        npt.assert_allclose(P_s, P_a)

    def test_repeated_updates_converge(self):
        """Multiple measurements of the same value should converge the estimate."""
        ukf = self._make_filter(d=1, alpha=1e-3)
        y_true = np.array([5.0])
        for _ in range(20):
            ukf.predict(omega=None, dt=1.0)
            ukf.update(y=y_true + np.random.default_rng(0).normal(0, 0.1, 1))
        state, _ = ukf.filter_state
        # Should be close to 5
        npt.assert_allclose(state[0], 5.0, atol=1.0)

    def test_covariance_remains_symmetric_pd(self):
        """Covariance matrix must stay symmetric and positive-definite."""
        ukf = self._make_filter(d=3)
        rng = np.random.default_rng(42)
        for _ in range(10):
            ukf.predict(omega=None, dt=0.5)
            ukf.update(y=rng.normal(0, 1, 3))
        _, P = ukf.filter_state
        npt.assert_allclose(P, P.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(P)
        self.assertTrue(np.all(eigvals > 0))


class TestUKFOnManifoldsSO2(unittest.TestCase):
    """Test UKF-M on the circle SO(2) ~ angle."""

    def _make_filter(self):
        Q = np.array([[0.01]])  # 1×1 noise
        R = np.array([[0.1]])
        state0 = 0.0  # scalar angle
        P0 = np.array([[0.5]])

        def f(s, omega, w, dt):  # pylint: disable=unused-argument
            # constant-velocity on SO(2)
            return s + w[0]

        def h(s):
            return np.array([s])

        return UKFOnManifolds(
            f=f, h=h,
            phi=_phi_so2, phi_inv=_phi_inv_so2,
            Q=Q, R=R,
            alpha=1e-3,
            state0=state0, P0=P0,
        )

    def test_initialization(self):
        ukf = self._make_filter()
        state, P = ukf.filter_state
        self.assertAlmostEqual(float(state), 0.0)
        npt.assert_array_equal(P, np.array([[0.5]]))

    def test_update_moves_toward_measurement(self):
        ukf = self._make_filter()
        ukf.predict(omega=None, dt=1.0)
        y = np.array([0.3])
        state_before, _ = ukf.filter_state
        ukf.update(y=y)
        state_after, _ = ukf.filter_state
        self.assertGreater(float(state_after), float(state_before))
        self.assertLess(float(state_after), float(y[0]))

    def test_covariance_decreases_after_update(self):
        ukf = self._make_filter()
        ukf.predict(omega=None, dt=1.0)
        _, P_pred = ukf.filter_state
        ukf.update(y=np.array([0.1]))
        _, P_upd = ukf.filter_state
        self.assertLess(float(P_upd[0, 0]), float(P_pred[0, 0]))


class TestUKFOnManifoldsLinearEquivalence(unittest.TestCase):
    """
    For a linear Gaussian model on Euclidean space, UKF-M should give results
    very close to the exact Kalman filter.
    """

    def test_linear_1d_matches_kalman(self):
        """
        x_{k+1} = x_k + w_k,  w_k ~ N(0, Q)
        y_k     = x_k   + v_k, v_k ~ N(0, R)
        """
        Q_val = 0.1
        R_val = 0.5
        x0 = 0.0
        P0 = 1.0
        y = 2.0

        # --- UKF-M ---
        def f(s, omega, w, dt):  # pylint: disable=unused-argument
            return s + w

        def h(s):
            return s

        ukf = UKFOnManifolds(
            f=f, h=h,
            phi=lambda s, xi: s + xi[0],
            phi_inv=lambda s_ref, s: np.array([s - s_ref]),
            Q=np.array([[Q_val]]),
            R=np.array([[R_val]]),
            alpha=1e-3,
            state0=np.array([x0]),
            P0=np.array([[P0]]),
        )
        ukf.predict(omega=None, dt=1.0)
        ukf.update(y=np.array([y]))
        state_ukfm, P_ukfm = ukf.filter_state

        # --- Exact Kalman ---
        P_pred = P0 + Q_val
        K = P_pred / (P_pred + R_val)
        x_upd = x0 + K * (y - x0)
        P_upd = (1 - K) * P_pred

        npt.assert_allclose(float(state_ukfm[0]), x_upd, rtol=1e-5)
        npt.assert_allclose(float(P_ukfm[0, 0]), P_upd, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
