from pyrecest.backend import array, diag
from pyrecest.filters import KalmanFilter


class KalmanFilterBenchmarks:
    def setup(self):
        self.system_matrix = array([[1.0, 1.0], [0.0, 1.0]])
        self.measurement_matrix = array([[1.0, 0.0]])
        self.system_noise_cov = diag(array([0.05, 0.01]))
        self.measurement_noise_cov = array([[0.25]])
        self.measurement = array([1.0])

    def time_predict_update_loop(self):
        filt = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))
        for _ in range(100):
            filt.predict_linear(self.system_matrix, self.system_noise_cov)
            filt.update_linear(
                self.measurement, self.measurement_matrix, self.measurement_noise_cov
            )
