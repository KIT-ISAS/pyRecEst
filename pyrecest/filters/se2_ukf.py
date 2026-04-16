"""SE(2) Unscented Kalman Filter.

Reference:
    A Stochastic Filter for Planar Rigid-Body Motions,
    Igor Gilitschenski, Gerhard Kurz, and Uwe D. Hanebeck,
    Proceedings of the 2015 IEEE International Conference on Multisensor
    Fusion and Integration for Intelligent Systems (MFI),
    San Diego, USA, 2015.
"""

# pylint: disable=redefined-builtin
from pyrecest.backend import (
    abs,
    array,
    asarray,
    concatenate,
    empty,
    eye,
    hstack,
    linalg,
    mean,
    vstack,
    zeros,
)
from pyrecest.distributions import GaussianDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import SE2FilterMixin


def _dual_quaternion_multiply(dq1, dq2):
    """Multiply two SE(2) dual quaternions.

    The compact dual quaternion for an SE(2) element is
    ``[cos(θ/2), sin(θ/2), d1, d2]`` where ``(d1, d2)`` are the dual
    parts encoding the 2-D translation.

    This is equivalent to computing the matrix product
    ``M(dq1) @ M(dq2)`` and extracting the result, where::

        M(dq) = [ dq[0]  dq[1]    0      0   ]
                [-dq[1]  dq[0]    0      0   ]
                [-dq[2]  dq[3] dq[0] -dq[1] ]
                [-dq[3] -dq[2] dq[1]  dq[0] ]

    Parameters
    ----------
    dq1 : array-like, shape (4,)
    dq2 : array-like, shape (4,)

    Returns
    -------
    array, shape (4,)
    """
    a, b, c, d = dq1
    e, f, g, h = dq2
    return array(
        [
            a * e - b * f,
            b * e + a * f,
            c * e + d * f + a * g - b * h,
            d * e - c * f + b * g + a * h,
        ]
    )


class SE2UKF(AbstractFilter, SE2FilterMixin):
    """Unscented Kalman Filter for planar rigid-body motions (SE(2)).

    The filter state is a :class:`~pyrecest.distributions.GaussianDistribution`
    over the 4-D **dual-quaternion** (compact) representation of SE(2):
    ``[cos(θ/2), sin(θ/2), d1, d2]``.  The first two components lie on
    the unit circle (quaternion part), and the last two are the dual parts
    encoding the 2-D translation.

    Use :meth:`~pyrecest.distributions.AbstractSE2Distribution.dual_quaternion_to_angle_pos`
    to convert the mean to an angle/position pair.

    Reference:
        Igor Gilitschenski, Gerhard Kurz, and Uwe D. Hanebeck,
        "A Stochastic Filter for Planar Rigid-Body Motions",
        Proc. IEEE MFI, San Diego, USA, 2015.
    """

    def __init__(self):
        # Default initial state: identity transform, isotropic uncertainty.
        initial_state = GaussianDistribution(
            array([1.0, 0.0, 0.0, 0.0]), eye(4)
        )
        SE2FilterMixin.__init__(self)
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
        ), "State must be a GaussianDistribution"
        assert new_state.C.shape == (4, 4), "Covariance must be 4×4"
        assert new_state.mu.shape[0] == 4, "Mean must be 4-D"
        assert (
            abs(linalg.norm(asarray(new_state.mu[:2])) - 1.0)
            < 1e-10
        ), "First two entries of the mean must be normalized"
        self._filter_state = new_state

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_identity(self, gauss_sys: GaussianDistribution):
        """Predict using a zero-mean identity motion model with noise.

        The model assumes:

            x_{t+1} = v [⊗] x_t

        where ``v`` is the system noise (zero mean in the manifold sense,
        i.e. ``gauss_sys.mu = [1, 0, 0, 0]``).

        Parameters
        ----------
        gauss_sys : GaussianDistribution
            4-D system noise distribution.  The mean must have its first
            two entries normalized (unit dual quaternion representing the
            identity element).
        """
        assert isinstance(gauss_sys, GaussianDistribution), (
            "gauss_sys must be a GaussianDistribution"
        )
        assert gauss_sys.C.shape == (4, 4), "System covariance must be 4×4"
        assert gauss_sys.mu.shape[0] == 4, "System noise mean must be 4-D"
        assert (
            abs(linalg.norm(asarray(gauss_sys.mu[:2])) - 1.0)
            < 1e-10
        ), "First two entries of the system noise mean must be normalized"

        mu = asarray(self._filter_state.mu)
        C = asarray(self._filter_state.C)
        mu_noise = asarray(gauss_sys.mu)
        C_noise = asarray(gauss_sys.C)

        # --- State sigma points (9 points) ---
        # MATLAB: [0, 2*chol(C)', -2*chol(C)'] + mu
        # chol(C) returns upper-triangular R with R'*R = C; columns of 2*R'
        # = columns of 2*L where L = lower-triangular Cholesky.
        L_state = linalg.cholesky(C)
        state_sigmas = empty((4, 9))
        state_sigmas[:, 0] = mu
        for i in range(4):
            state_sigmas[:, i + 1] = mu + 2.0 * L_state[:, i]
            state_sigmas[:, i + 5] = mu - 2.0 * L_state[:, i]

        # Normalise quaternion part of state sigma points
        norms = linalg.norm(state_sigmas[:2, :], axis=0)
        state_sigmas[:2, :] /= norms[None, :]

        # --- Noise sigma points (9 points) ---
        # MATLAB: [0, chol(4*C_noise)', -chol(4*C_noise)'] + mu_noise
        L_noise = linalg.cholesky(4.0 * C_noise)
        noise_sigmas = empty((4, 9))
        noise_sigmas[:, 0] = mu_noise
        for i in range(4):
            noise_sigmas[:, i + 1] = mu_noise + L_noise[:, i]
            noise_sigmas[:, i + 5] = mu_noise - L_noise[:, i]

        # Normalise quaternion part of noise sigma points
        norms = linalg.norm(noise_sigmas[:2, :], axis=0)
        noise_sigmas[:2, :] /= norms[None, :]

        # --- Prediction samples (81 = 9×9 samples) ---
        pred_samples = empty((4, 81))
        for i in range(9):
            for j in range(9):
                pred_samples[:, i * 9 + j] = _dual_quaternion_multiply(
                    state_sigmas[:, i], noise_sigmas[:, j]
                )

        # Covariance from prediction samples (not mean-centred, like MATLAB)
        CP = (pred_samples @ pred_samples.T) / 81.0
        new_C = (CP + CP.T) / 2.0  # symmetrise to avoid numerical issues

        # Mean from state sigma points (normalised)
        new_mu = mean(state_sigmas, axis=1)
        new_mu[:2] /= linalg.norm(new_mu[:2])

        self._filter_state = GaussianDistribution(
            new_mu, new_C, check_validity=False
        )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_identity(self, gauss_meas: GaussianDistribution, z):
        """Incorporate a dual-quaternion measurement.

        Assumes the identity measurement model  ``z = x [⊗] v``  where
        ``v`` is additive dual-quaternion noise with distribution
        *gauss_meas*.

        Parameters
        ----------
        gauss_meas : GaussianDistribution
            4-D measurement noise distribution.  The mean must have its
            first two entries normalised.
        z : array-like, shape (4,)
            Measurement given as a compact dual quaternion.
        """
        assert isinstance(gauss_meas, GaussianDistribution), (
            "gauss_meas must be a GaussianDistribution"
        )
        assert gauss_meas.C.shape == (4, 4), "Measurement covariance must be 4×4"
        assert gauss_meas.mu.shape[0] == 4, "Measurement noise mean must be 4-D"
        assert (
            abs(linalg.norm(asarray(gauss_meas.mu[:2])) - 1.0)
            < 1e-10
        ), "First two entries of the measurement noise mean must be normalised"

        mu = asarray(self._filter_state.mu)
        C = asarray(self._filter_state.C)
        mu_noise = asarray(gauss_meas.mu)
        C_noise = asarray(gauss_meas.C)
        z = asarray(z).flatten()

        # Take the closer of the two antipodal representations
        if linalg.norm(z - mu) > linalg.norm(-z - mu):
            z = -z

        # --- Augmented state and covariance ---
        xaug = concatenate([mu, mu_noise])
        CAUG = vstack([hstack([C, zeros((4, 4))]), hstack([zeros((4, 4)), C_noise])])

        # 17 sigma points from augmented state
        # MATLAB: [zeros(8,1), chol(8*CAUG)', -chol(8*CAUG)'] + xaug
        L_aug = linalg.cholesky(8.0 * CAUG)
        aug_sigmas = empty((8, 17))
        aug_sigmas[:, 0] = xaug
        for i in range(8):
            aug_sigmas[:, i + 1] = xaug + L_aug[:, i]
            aug_sigmas[:, i + 9] = xaug - L_aug[:, i]

        # Normalise quaternion part of the noise sigma points (indices 4:6)
        norms_noise = linalg.norm(aug_sigmas[4:6, :], axis=0)
        norm_noise_rot = aug_sigmas[4:6, :] / norms_noise[None, :]
        norm_noise = vstack([norm_noise_rot, aug_sigmas[6:8, :]])

        # --- Apply measurement function: z_pred = state_part [⊗] noise ---
        meas_samples = empty((4, 17))
        for i in range(17):
            meas_samples[:, i] = _dual_quaternion_multiply(
                aug_sigmas[:4, i], norm_noise[:, i]
            )

        # --- UKF cross-covariance and measurement covariance ---
        mean_meas = mean(meas_samples, axis=1)
        meas_dev = meas_samples - mean_meas[:, None]

        # PXY = (1/17) * aug_sigmas @ meas_dev^T
        PXY = (aug_sigmas @ meas_dev.T) / 17.0

        # PY = (1/17) * meas_dev @ meas_dev^T  (biased, mean already removed)
        PY = (meas_dev @ meas_dev.T) / 17.0

        # --- Kalman update ---
        K = PXY @ linalg.inv(PY)
        xeaug = xaug + K @ (z - mean_meas)
        xe = xeaug[:4]
        CEAUG = CAUG - K @ PY @ K.T
        new_C = CEAUG[:4, :4]

        # Normalise quaternion part of updated mean
        xe[:2] /= linalg.norm(xe[:2])

        self._filter_state = GaussianDistribution(
            xe, new_C, check_validity=False
        )

    # ------------------------------------------------------------------
    # Point estimate
    # ------------------------------------------------------------------

    def get_point_estimate(self):
        """Return the mean of the current state estimate.

        The returned value is a 4-D dual quaternion
        ``[cos(θ/2), sin(θ/2), d1, d2]``.  Use
        :meth:`~pyrecest.distributions.AbstractSE2Distribution.dual_quaternion_to_angle_pos`
        to convert to an angle/position representation.
        """
        return self._filter_state.mu
