import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from abstract_euclidean_filter import AbstractEuclideanFilter
from gaussian_distribution import GaussianDistribution

class KalmanFilter(AbstractEuclideanFilter):
    def __init__(self, prior_mean, prior_cov):
        self.kf = FilterPyKalmanFilter(dim_x=prior_mean.__len__(), dim_z=prior_mean.__len__()) # Set it identical to the dimensionality of the state because we do not know yet.
        self.set_state(prior_mean, prior_cov)

    def set_state(self, mean, cov):
        self.kf.x = np.asarray(mean)
        self.kf.P = np.asarray(cov) # FilterPy uses .P

    def predict_identity(self, sys_noise_mean, sys_noise_cov):
        self.kf.predict(Q=sys_noise_cov, u=sys_noise_mean)

    def predict_linear(self, system_matrix, sys_noise_cov, input=None):
        if input is None:
            B = None
        else:
            B = np.eye(input.shape[0])
        self.kf.predict(F=system_matrix, Q=sys_noise_cov, B=B, u=input)

    def update_identity(self, meas, meas_cov):
        self.kf.update(z=np.array([meas]), R=meas_cov, H=np.eye(self.kf.x.shape[0]))

    def update_linear(self, measurement, measurement_matrix, cov_mat_meas):
        self.kf.update(z=measurement, R=cov_mat_meas, H=measurement_matrix)

    def get_estimate(self):
        return GaussianDistribution(self.kf.x, self.kf.P)
