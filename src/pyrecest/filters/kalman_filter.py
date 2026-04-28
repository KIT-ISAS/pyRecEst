# pylint: disable=no-name-in-module,no-member

from pyrecest.backend import atleast_1d, atleast_2d, eye
from pyrecest.distributions import GaussianDistribution

from ._linear_gaussian import linear_gaussian_predict, linear_gaussian_update
from .abstract_filter import AbstractFilter
from .manifold_mixins import EuclideanFilterMixin


class KalmanFilter(AbstractFilter, EuclideanFilterMixin):
    """Kalman filter for linear Gaussian Euclidean state-space models.

    The filter state is stored as a :class:`GaussianDistribution` with mean
    vector shape ``(n,)`` and covariance matrix shape ``(n, n)``. Prediction
    uses ``x_k = F x_{k-1} + u + w`` and updates use
    ``z_k = H x_k + v``.
    """

    def __init__(self, initial_state):
        """Create a Kalman filter from a Gaussian state.

        Parameters
        ----------
        initial_state : GaussianDistribution or tuple
            Initial state distribution. Tuples must contain ``(mean,
            covariance)`` with mean shape ``(n,)`` or scalar shape for a
            one-dimensional state, and covariance shape ``(n, n)``.
        """
        EuclideanFilterMixin.__init__(self)
        AbstractFilter.__init__(self, self._coerce_state(initial_state))

    @staticmethod
    def _coerce_state(state):
        if isinstance(state, GaussianDistribution):
            return GaussianDistribution(
                atleast_1d(state.mu),
                atleast_2d(state.C),
                check_validity=False,
            )
        if isinstance(state, tuple) and len(state) == 2:
            mean, covariance = state
            return GaussianDistribution(
                atleast_1d(mean),
                atleast_2d(covariance),
                check_validity=False,
            )
        raise ValueError(
            "state must be a GaussianDistribution or a tuple of (mean, covariance)"
        )

    @property
    def dim(self):
        """Return the dimension ``n`` of the Euclidean state vector."""
        return self._filter_state.dim

    @property
    def filter_state(self) -> GaussianDistribution:
        """Return a Gaussian copy of the current filter state."""
        return GaussianDistribution(
            self._filter_state.mu,
            self._filter_state.C,
            check_validity=False,
        )

    @filter_state.setter
    def filter_state(self, new_state):
        """
        Set the filter state.

        Parameters
        ----------
        new_state : GaussianDistribution or tuple
            Replacement state distribution. Tuples must contain ``(mean,
            covariance)``.
        """
        self._filter_state = self._coerce_state(new_state)

    def predict_identity(self, sys_noise_cov, sys_input=None):
        """Predict one step with an identity transition matrix.

        Parameters
        ----------
        sys_noise_cov : array-like, shape (n, n)
            Additive process-noise covariance.
        sys_input : array-like, shape (n,), optional
            Additive deterministic input applied after the identity transition.
        """
        self.predict_linear(eye(self.dim), sys_noise_cov, sys_input)

    def predict_linear(
        self,
        system_matrix,
        sys_noise_cov,
        sys_input=None,
    ):
        """Predict one step with a linear Gaussian system model.

        Parameters
        ----------
        system_matrix : array-like, shape (n, n)
            State-transition matrix ``F``.
        sys_noise_cov : array-like, shape (n, n)
            Additive process-noise covariance ``Q``.
        sys_input : array-like, shape (n,), optional
            Additive deterministic input ``u``.
        """
        new_mean, new_covariance = linear_gaussian_predict(
            mean=self._filter_state.mu,
            covariance=self._filter_state.C,
            system_matrix=system_matrix,
            sys_noise_cov=sys_noise_cov,
            sys_input=sys_input,
        )
        self._filter_state = GaussianDistribution(
            new_mean,
            new_covariance,
            check_validity=False,
        )

    def update_identity(self, meas_noise, measurement):
        """Update with a measurement matrix equal to the identity.

        Parameters
        ----------
        meas_noise : array-like, shape (n, n)
            Measurement-noise covariance.
        measurement : array-like, shape (n,)
            Measurement vector.
        """
        self.update_linear(
            measurement=measurement,
            measurement_matrix=eye(self.dim),
            meas_noise=meas_noise,
        )

    def update_linear(
        self,
        measurement,
        measurement_matrix,
        meas_noise,
    ):
        """Update the state with a linear Gaussian measurement model.

        Parameters
        ----------
        measurement : array-like, shape (m,)
            Measurement vector ``z``.
        measurement_matrix : array-like, shape (m, n)
            Measurement matrix ``H`` mapping state vectors to measurement
            vectors.
        meas_noise : array-like, shape (m, m)
            Measurement-noise covariance ``R``.
        """
        new_mean, new_covariance = linear_gaussian_update(
            mean=self._filter_state.mu,
            covariance=self._filter_state.C,
            measurement=measurement,
            measurement_matrix=measurement_matrix,
            meas_noise=meas_noise,
        )
        self._filter_state = GaussianDistribution(
            new_mean,
            new_covariance,
            check_validity=False,
        )

    def get_point_estimate(self):
        """Return the posterior mean vector with shape ``(n,)``."""
        return self._filter_state.mu
