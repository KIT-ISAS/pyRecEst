import copy
import unittest

import numpy as np
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.kalman_filter import KalmanFilter


class KalmanFilterTest(unittest.TestCase):
    def test_initialization_mean_cov(self):
        filter_custom = KalmanFilter((np.array([1]), np.array([[10000]])))
        self.assertEqual(filter_custom.get_point_estimate(), [1])

    def test_initialization_gauss(self):
        filter_custom = KalmanFilter(
            initial_state=GaussianDistribution(np.array([4]), np.array([[10000]]))
        )
        self.assertEqual(filter_custom.get_point_estimate(), [4])

    def test_update_with_likelihood_1d(self):
        kf = KalmanFilter((np.array([0]), np.array([[1]])))
        kf.update_identity(np.array(3), np.array(1))
        self.assertEqual(kf.get_point_estimate(), 1.5)

    def test_update_with_meas_noise_and_meas_1d(self):
        kf = KalmanFilter((np.array([0]), np.array([[1]])))
        kf.update_identity(np.array(4), np.array(1))
        self.assertEqual(kf.get_estimate().C, 0.5)
        self.assertEqual(kf.get_point_estimate(), 2)

    def test_update_linear_2d(self):
        filter_add = KalmanFilter((np.array([0, 1]), np.diag([1, 2])))
        filter_id = copy.deepcopy(filter_add)
        gauss = GaussianDistribution(np.array([1, 0]), np.diag([2, 1]))
        filter_add.update_linear(gauss.mu, np.eye(2), gauss.C)
        filter_id.update_identity(gauss.mu, gauss.C)
        self.assertTrue(
            np.allclose(filter_add.get_point_estimate(), filter_id.get_point_estimate())
        )
        self.assertTrue(
            np.allclose(filter_add.filter_state.C, filter_id.filter_state.C)
        )

    def test_predict_identity_1d(self):
        kf = KalmanFilter((np.array([0]), np.array([[1]])))
        kf.predict_identity(np.array(0), np.array(3))
        self.assertEqual(kf.get_point_estimate(), 0)
        self.assertEqual(kf.filter_state.C, 4)

    def test_predict_linear_2d(self):
        kf = KalmanFilter((np.array([0, 1]), np.diag([1, 2])))
        kf.predict_linear(np.diag([1, 2]), np.diag([2, 1]))
        self.assertTrue(np.allclose(kf.get_point_estimate(), np.array([0, 2])))
        self.assertTrue(np.allclose(kf.filter_state.C, np.diag([3, 9])))
        kf.predict_linear(np.diag([1, 2]), np.diag([2, 1]), np.array([2, -2]))
        self.assertTrue(np.allclose(kf.get_point_estimate(), np.array([2, 2])))


if __name__ == "__main__":
    unittest.main()
