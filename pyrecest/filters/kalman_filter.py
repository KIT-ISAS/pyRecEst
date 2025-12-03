# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from bayesian_filters.kalman import KalmanFilter as BayesianFiltersKalmanFilter

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import eye
from pyrecest.distributions import GaussianDistribution

from .abstract_euclidean_filter import AbstractEuclideanFilter


class KalmanFilter(AbstractEuclideanFilter):
    def __init__(self, initial_state):
        """
        Initialize the Kalman filter with the initial state.

        :param initial_state: Provide GaussianDistribution or mean and covariance as initial state.
        """
        if isinstance(initial_state, GaussianDistribution):
            dim_x = initial_state.dim
        elif isinstance(initial_state, tuple) and len(initial_state) == 2:
            dim_x = len(initial_state[0])
        else:
            raise ValueError(
                "initial_state must be a GaussianDistribution or a tuple of (mean, covariance)"
            )

        self._filter_state = BayesianFiltersKalmanFilter(dim_x=dim_x, dim_z=dim_x)
        self.filter_state = initial_state

    @property
    def dim(self):
        """Returns the dimension of the state."""
        return self._filter_state.x.shape[0]

    @property
    def filter_state(
        self,
    ) -> GaussianDistribution:
        return GaussianDistribution(self._filter_state.x, self._filter_state.P)
    
    @filter_state.setter
    def filter_state(self, new_state):
        """
        Set the filter state.

        :param new_state: Provide GaussianDistribution or mean and covariance as state.
        """
        if isinstance(new_state, GaussianDistribution):
            self._filter_state.x = new_state.mu
            self._filter_state.P = new_state.C
        elif isinstance(new_state, tuple) and len(new_state) == 2:
            self._filter_state.x = new_state[0]
            self._filter_state.P = new_state[1]
        else:
            raise ValueError(
                "new_state must be a GaussianDistribution or a tuple of (mean, covariance)"
            )

    def predict_identity(self, sys_noise_cov, sys_input=None):
        """
        Predicts the next state assuming identity transition matrix.

        :param sys_noise_mean: System noise mean.
        :param sys_input: System noise covariance.
        """
        system_matrix = eye(self._filter_state.x.shape[0])
        B = eye(system_matrix.shape[0]) if sys_input is not None else None
        self._filter_state.predict(F=system_matrix, Q=sys_noise_cov, B=B, u=sys_input)

    def predict_linear(
        self,
        system_matrix,
        sys_noise_cov,
        sys_input=None,
    ):
        """
        Predicts the next state assuming a linear system model.

        :param system_matrix: System transition matrix.
        :param sys_noise_cov: System noise covariance.
        :param sys_input: System input.
        """
        assert (
            pyrecest.backend.__backend_name__ != "pytorch"
        ), "Not supported on this backend"
        if sys_input is not None and system_matrix.shape[0] != sys_input.shape[0]:
            raise ValueError(
                "The number of rows in system_matrix should match the number of elements in sys_input"
            )

        B = eye(system_matrix.shape[0]) if sys_input is not None else None
        self._filter_state.predict(F=system_matrix, Q=sys_noise_cov, B=B, u=sys_input)

    def update_identity(self, meas_noise, measurement):
        """
        Update the filter state with measurement, assuming identity measurement matrix.

        :param measurement: Measurement.
        :param meas_noise_cov: Measurement noise covariance.
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
        """
        Update the filter state with measurement, assuming a linear measurement model.

        :param measurement: Measurement.
        :param measurement_matrix: Measurement matrix.
        :param meas_noise: Covariance matrix for measurement.
        """
        assert (
            pyrecest.backend.__backend_name__ != "pytorch"
        ), "Not supported on this backend"
        self._filter_state.dim_z = measurement_matrix.shape[0]
        self._filter_state.update(z=measurement, R=meas_noise, H=measurement_matrix)

    def get_point_estimate(self):
        """Returns the mean of the current filter state."""
        return self._filter_state.x
