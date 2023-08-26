import numpy as np

from .abstract_extended_object_tracker import AbstractExtendedObjectTracker


class RandomMatrixTracker(AbstractExtendedObjectTracker):
    def __init__(self, initial_state, initial_covariance, initial_extent, **kwargs):
        super().__init__(**kwargs)
        self.state = np.array(initial_state)  # Initial kinematic state
        self.covariance = np.array(
            initial_covariance
        )  # Initial state covariance matrix
        self.extent = np.array(initial_extent)  # Initial extent matrix
        self.alpha = 0

    def get_point_estimate(self):
        # Combines the kinematic state and flattened extent matrix into one vector
        return np.concatenate([self.state, self.extent.flatten()])

    def get_point_estimate_kinematics(self):
        # Returns just the kinematic state
        return self.state

    def get_point_estimate_extent(self, flatten_matrix=False):
        # Returns just the extent matrix
        if flatten_matrix:
            return self.extent.flatten()
        return self.extent

    def predict(self, dt, Cw, tau, system_matrix):
        F = system_matrix
        x_rows = self.state.shape[0]
        y_rows = x_rows // 2

        if np.isscalar(Cw):
            Cw = Cw * np.eye(x_rows)

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Cw

        self.alpha = y_rows + np.exp(-dt / tau) * (self.alpha - y_rows)

    # pylint: disable=too-many-locals
    def update(self, measurements, meas_mat, meas_noise_cov):
        Cv = meas_noise_cov
        ys = measurements
        H = meas_mat

        y_rows, y_cols = ys.shape
        if y_cols < y_rows + 1:
            raise ValueError("Too few measurements.")

        y_ = np.mean(ys, axis=1, keepdims=True)
        ys_demean = ys - y_
        Y_ = ys_demean @ ys_demean.T

        Hx = H @ self.state

        Y = self.extent + Cv
        S = H @ self.covariance @ H.T + Y / y_cols
        K = self.covariance @ np.linalg.solve(S, H).T
        self.state = self.state + K @ (y_.flatten() - Hx)
        self.covariance = self.covariance - K @ S @ K.T

        Xsqrt = np.linalg.cholesky(self.extent)
        Ssqrt = np.linalg.cholesky(S)
        Ysqrt = np.linalg.cholesky(Y)

        Nsqrt = Xsqrt * np.linalg.inv(Ssqrt) @ (y_ - Hx)
        N = Nsqrt @ Nsqrt.T
        XYsqrt = Xsqrt * np.linalg.inv(Ysqrt)

        self.extent = (self.alpha * self.extent + N + XYsqrt @ Y_ @ XYsqrt.T) / (
            self.alpha + y_cols
        )

        self.alpha = self.alpha + y_cols
