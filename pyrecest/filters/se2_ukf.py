"""
SE(2) Unscented Kalman Filter using dual-quaternion state representation.

Reference:
    A Stochastic Filter for Planar Rigid-Body Motions,
    Igor Gilitschenski, Gerhard Kurz, and Uwe D. Hanebeck,
    Proceedings of the 2015 IEEE International Conference on Multisensor
    Fusion and Integration for Intelligent Systems (MFI),
    San Diego, USA, 2015.
"""

# pylint: disable=no-name-in-module,no-member
import numpy as np
from scipy.linalg import cholesky

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import SE2FilterMixin


def _dual_quaternion_multiply(dq1, dq2):
    """Multiply two SE(2) dual quaternions.

    Each dual quaternion is represented as a length-4 array
    ``[q1, q2, d1, d2]`` where ``[q1, q2]`` is the unit-norm
    rotation part and ``[d1, d2]`` is the dual (translation) part.

    The formula is derived from the 4×4 matrix representation used in
    the libDirectional SE2 class::

        M(dq) = [[ q1,  q2,  0,  0 ],
                 [-q2,  q1,  0,  0 ],
                 [-d1,  d2, q1, -q2],
                 [-d2, -d1, q2,  q1]]

    and ``dq_product = M(dq1) @ M(dq2)``.

    Parameters
    ----------
    dq1, dq2 : array_like, shape (4,)
        SE(2) dual quaternions.

    Returns
    -------
    np.ndarray, shape (4,)
        Product dual quaternion.
    """
    a, b, c, d = dq1[0], dq1[1], dq1[2], dq1[3]
    e, f, g, h = dq2[0], dq2[1], dq2[2], dq2[3]
    return np.array(
        [
            a * e - b * f,
            b * e + a * f,
            c * e + d * f + a * g - b * h,
            d * e - c * f + b * g + a * h,
        ]
    )


class SE2UKF(AbstractFilter, SE2FilterMixin):
    """Unscented Kalman Filter for planar rigid-body motion on SE(2).

    The state is represented as a :class:`~pyrecest.distributions.GaussianDistribution`
    over the 4-D dual-quaternion embedding of SE(2).  The first two
    entries of the mean encode the rotation (and must satisfy
    ``||mu[0:2]|| == 1``); the last two entries encode the translation.

    Reference:
        A Stochastic Filter for Planar Rigid-Body Motions,
        Igor Gilitschenski, Gerhard Kurz, and Uwe D. Hanebeck,
        IEEE MFI 2015.
    """

    def __init__(self):
        # Initialise with a trivial identity-like state.
        initial_state = GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]),
            array(
                [
                    [0.25, 0.0, 0.0, 0.0],
                    [0.0, 0.25, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
        AbstractFilter.__init__(self, initial_state)

    # ------------------------------------------------------------------
    # filter_state property
    # ------------------------------------------------------------------

    @property
    def filter_state(self) -> GaussianDistribution:
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state: GaussianDistribution):
        assert isinstance(
            new_state, GaussianDistribution
        ), "filter_state must be a GaussianDistribution"
        mu = np.asarray(new_state.mu)
        assert mu.shape == (4,), "Mean must be a 4-D vector."
        assert np.isclose(
            np.linalg.norm(mu[0:2]), 1.0
        ), "First two entries of the mean must be normalised."
        assert np.asarray(new_state.C).shape == (
            4,
            4,
        ), "Covariance must be 4×4."
        self._filter_state = new_state

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_identity(self, gauss_sys: GaussianDistribution):
        """Predict with a left-multiplicative noise model on SE(2).

        The motion model is::

            x_{t+1} = v [⊕] x_t

        where ``v`` is the system noise.  The mean of ``gauss_sys``
        encodes the (dual-quaternion) noise mean; the noise is assumed
        zero-mean in the manifold sense.

        Parameters
        ----------
        gauss_sys : GaussianDistribution
            System noise distribution.  Must have a 4-D mean (first two
            entries normalised) and a 4×4 covariance.
        """
        assert isinstance(gauss_sys, GaussianDistribution)
        mu_sys = np.asarray(gauss_sys.mu, dtype=float)
        C_sys = np.asarray(gauss_sys.C, dtype=float)
        assert mu_sys.shape == (4,), "System noise mean must be 4-D."
        assert C_sys.shape == (4, 4), "System noise covariance must be 4×4."
        assert np.isclose(
            np.linalg.norm(mu_sys[0:2]), 1.0
        ), "First two entries of the system noise mean must be normalised."

        mu = np.asarray(self._filter_state.mu, dtype=float)
        C = np.asarray(self._filter_state.C, dtype=float)

        # --- State sigma points: [mu, mu ± 2*L[:,k]] for k=0..3 (9 pts) ---
        L = cholesky(C, lower=True)  # 4×4 lower-triangular
        cols = [mu]
        for k in range(4):
            cols.append(mu + 2.0 * L[:, k])
        for k in range(4):
            cols.append(mu - 2.0 * L[:, k])
        state_samples = np.column_stack(cols)  # 4×9

        # Normalise rotation part of state sigma points.
        norms = np.linalg.norm(state_samples[0:2, :], axis=0)
        state_samples[0:2, :] /= norms[np.newaxis, :]

        # --- Noise sigma points: [mu_sys, mu_sys ± L_n[:,k]] for k=0..3 (9 pts) ---
        L_n = cholesky(4.0 * C_sys, lower=True)  # 4×4
        n_cols = [mu_sys]
        for k in range(4):
            n_cols.append(mu_sys + L_n[:, k])
        for k in range(4):
            n_cols.append(mu_sys - L_n[:, k])
        noise_samples = np.column_stack(n_cols)  # 4×9

        # Normalise rotation part of noise sigma points.
        norms = np.linalg.norm(noise_samples[0:2, :], axis=0)
        noise_samples[0:2, :] /= norms[np.newaxis, :]

        # --- Predicted samples: all 81 = 9×9 combinations ---
        pred_samples = np.empty((4, 81))
        for i in range(9):
            for j in range(9):
                pred_samples[:, i * 9 + j] = _dual_quaternion_multiply(
                    state_samples[:, i], noise_samples[:, j]
                )

        # Covariance from outer-product mean (not centred – matches MATLAB).
        CP = pred_samples @ pred_samples.T / 81.0
        new_C = (CP + CP.T) / 2.0  # symmetrise

        # Mean from state samples (not prediction samples – matches MATLAB).
        new_mu = np.mean(state_samples, axis=1)
        new_mu[0:2] /= np.linalg.norm(new_mu[0:2])

        self._filter_state = GaussianDistribution(array(new_mu), array(new_C))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_identity(self, gauss_meas: GaussianDistribution, z):
        """Incorporate a dual-quaternion measurement.

        The measurement model is::

            z = x [⊕] v

        where ``v`` is the measurement noise.

        Parameters
        ----------
        gauss_meas : GaussianDistribution
            Measurement noise distribution.  Must have a 4-D mean (first
            two entries normalised) and a 4×4 covariance.
        z : array_like, shape (4,)
            Measurement in dual-quaternion representation.
        """
        assert isinstance(gauss_meas, GaussianDistribution)
        mu_meas = np.asarray(gauss_meas.mu, dtype=float)
        C_meas = np.asarray(gauss_meas.C, dtype=float)
        assert mu_meas.shape == (4,), "Measurement noise mean must be 4-D."
        assert C_meas.shape == (4, 4), "Measurement noise covariance must be 4×4."
        assert np.isclose(
            np.linalg.norm(mu_meas[0:2]), 1.0
        ), "First two entries of the measurement noise mean must be normalised."

        z = np.asarray(z, dtype=float).ravel()
        assert z.shape == (4,), "Measurement z must be a 4-D vector."

        mu = np.asarray(self._filter_state.mu, dtype=float)
        C = np.asarray(self._filter_state.C, dtype=float)

        # Take the closer antipodal representative.
        if np.linalg.norm(z - mu) > np.linalg.norm(-z - mu):
            z = -z

        # --- Augmented state and covariance ---
        x_aug = np.concatenate([mu, mu_meas])  # 8-D
        C_aug = np.block([[C, np.zeros((4, 4))], [np.zeros((4, 4)), C_meas]])  # 8×8

        # --- Augmented sigma points: [x_aug, x_aug ± L_aug[:,k]] for k=0..7 (17 pts) ---
        L_aug = cholesky(8.0 * C_aug, lower=True)  # 8×8
        aug_cols = [x_aug]
        for k in range(8):
            aug_cols.append(x_aug + L_aug[:, k])
        for k in range(8):
            aug_cols.append(x_aug - L_aug[:, k])
        aug_samples = np.column_stack(aug_cols)  # 8×17

        # Normalise rotation part of state sigma points (rows 0–1).
        norms = np.linalg.norm(aug_samples[0:2, :], axis=0)
        aug_samples[0:2, :] /= norms[np.newaxis, :]

        # Extract and normalise the noise-rotation part (rows 4–5).
        noise_rot = aug_samples[4:6, :].copy()
        norms_n = np.linalg.norm(noise_rot, axis=0)
        noise_rot /= norms_n[np.newaxis, :]
        # Build the full normalised noise vectors (4-D each column).
        norm_noise = np.vstack([noise_rot, aug_samples[6:8, :]])  # 4×17

        # --- Apply measurement function: z_i = state_i ⊕ noise_i ---
        meas_samples = np.empty((4, 17))
        for i in range(17):
            meas_samples[:, i] = _dual_quaternion_multiply(
                aug_samples[0:4, i], norm_noise[:, i]
            )

        # --- Covariance matrices ---
        meas_mean = np.mean(meas_samples, axis=1)
        meas_dev = meas_samples - meas_mean[:, np.newaxis]  # 4×17

        # Cross-covariance: (aug_samples - x_aug) * meas_dev' / 17
        # Because meas_dev sums to zero, using aug_samples directly is
        # equivalent to using centred aug_samples.
        cross = aug_samples @ meas_dev.T / 17.0  # 8×4 (P_XY)
        P_Y = meas_dev @ meas_dev.T / 17.0  # 4×4 (innovation covariance)

        # --- Kalman update ---
        K = cross @ np.linalg.inv(P_Y)  # 8×4
        x_aug_upd = x_aug + K @ (z - meas_mean)  # 8-D
        C_aug_upd = C_aug - K @ P_Y @ K.T  # 8×8

        new_mu = x_aug_upd[0:4]
        new_C = C_aug_upd[0:4, 0:4]
        new_C = (new_C + new_C.T) / 2.0  # symmetrise

        # Renormalise rotation part of the mean.
        new_mu[0:2] /= np.linalg.norm(new_mu[0:2])

        self._filter_state = GaussianDistribution(array(new_mu), array(new_C))

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------

    def get_point_estimate(self):
        """Return the mean of the current state estimate."""
        return self._filter_state.mu
