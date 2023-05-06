import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from pyrecest.distributions import GaussianDistribution

from .abstract_euclidean_filter import AbstractEuclideanFilter


class KalmanFilter(AbstractEuclideanFilter):
    def __init__(self, initial_state=None):
        """ Provide GaussianDistribution or mean and covariance as initial state."""
        if isinstance(initial_state, GaussianDistribution):
            dim_x = initial_state.dim
        else:
            assert len(initial_state) == 2
            dim_x = len(initial_state[0])

        self.kf = FilterPyKalmanFilter(
            dim_x=dim_x, dim_z=dim_x
        )  # Set dim_z identical to the dimensionality of the state because we do not know yet.
        self.set_state(initial_state)

    def set_state(self, state):
        """ Provide GaussianDistribution or mean and covariance as state."""
        
        if isinstance(state, GaussianDistribution):
            mean = state.mu
            cov = state.C
        else:
            assert len(state) == 2
            mean = state[0]
            cov = state[1]
            
        self.kf.x = np.asarray(mean)
        self.kf.P = np.asarray(cov)  # FilterPy uses .P

    def predict_identity(self, sys_noise_mean, sys_noise_cov):
        self.kf.predict(Q=sys_noise_cov, u=sys_noise_mean)

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None):
        if sys_input is None:
            B = None
        else:
            B = np.eye(sys_input.shape[0])
        self.kf.predict(F=system_matrix, Q=sys_noise_cov, B=B, u=sys_input)

    def update_identity(self, meas, meas_noise_cov):
        self.kf.update(
            z=np.array([meas]), R=meas_noise_cov, H=np.eye(self.kf.x.shape[0])
        )

    def update_linear(self, measurement, measurement_matrix, cov_mat_meas):
        self.kf.update(z=measurement, R=cov_mat_meas, H=measurement_matrix)

    def get_point_estimate(self):
        return self.kf.x

    def get_estimate(self):
        return GaussianDistribution(self.kf.x, self.kf.P)
