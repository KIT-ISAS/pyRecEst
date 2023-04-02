import numpy as np
from filterpy.kalman import KalmanFilter as FilterPyKalmanFilter
from abstract_euclidean_filter import AbstractEuclideanFilter
from gaussian_distribution import GaussianDistribution

class KalmanFilter(AbstractEuclideanFilter):
    def __init__(self, prior_mean = None, prior_cov = None, prior_gauss = None):
        if prior_gauss is not None:
            dim_x = prior_gauss.dim
        else:
            dim_x = prior_mean.__len__()
            
        self.kf = FilterPyKalmanFilter(dim_x=dim_x, dim_z=dim_x) # Set dim_z identical to the dimensionality of the state because we do not know yet.
        self.set_state(prior_mean, prior_cov, prior_gauss)

    def set_state(self, mean = None, cov = None, gauss = None):
        # Set state either by providing a GaussianDistribution or mean and covariance
        if gauss is None:
            assert mean is not None and cov is not None
        else:
            assert mean is None and cov is None
            mean = gauss.mu
            cov = gauss.C
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
