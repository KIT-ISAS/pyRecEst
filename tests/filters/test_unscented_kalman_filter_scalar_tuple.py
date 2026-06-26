import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter


class UnscentedKalmanFilterScalarTupleStateTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_initialization_accepts_scalar_tuple_state(self):
        kf = UnscentedKalmanFilter((1.0, 10000.0))

        npt.assert_allclose(kf.get_point_estimate(), array([1.0]))
        self.assertEqual(kf.filter_state.mu.shape, (1,))
        self.assertEqual(kf.filter_state.C.shape, (1, 1))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_filter_state_setter_accepts_scalar_tuple_state(self):
        kf = UnscentedKalmanFilter(GaussianDistribution(array([0.0]), array([[1.0]])))

        kf.filter_state = (2.0, 3.0)

        npt.assert_allclose(kf.filter_state.mu, array([2.0]))
        npt.assert_allclose(kf.filter_state.C, array([[3.0]]))
        kf.update_identity(meas_noise=array([[1.0]]), measurement=array([4.0]))
        npt.assert_allclose(kf.get_point_estimate(), array([3.5]))


if __name__ == "__main__":
    unittest.main()
