import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import abs, all, array, cos, linalg, sin
from pyrecest.distributions.hypersphere_subset.bingham_distribution import (
    BinghamDistribution,
)
from pyrecest.filters.bingham_filter import BinghamFilter


class TestBinghamFilter2D(unittest.TestCase):
    def setUp(self):
        Z = array([-5.0, 0.0])
        phi = 0.4
        M = array([[cos(phi), -sin(phi)], [sin(phi), cos(phi)]])
        self.B = BinghamDistribution(Z, M)
        self.filter = BinghamFilter()
        self.Bnoise = BinghamDistribution(
            array([-3.0, 0.0]), array([[0.0, 1.0], [1.0, 0.0]])
        )

    def test_set_state_and_get_estimate(self):
        self.filter.filter_state = self.B
        B1 = self.filter.filter_state
        self.assertIsInstance(B1, BinghamDistribution)
        npt.assert_array_equal(self.B.M, B1.M)
        npt.assert_array_equal(self.B.Z, B1.Z)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_identity(self):
        self.filter.filter_state = self.B
        self.filter.predict_identity(self.Bnoise)
        B2 = self.filter.filter_state
        self.assertIsInstance(B2, BinghamDistribution)
        # Each column of M is determined only up to sign
        npt.assert_allclose(abs(self.B.M), abs(B2.M), rtol=1e-5, atol=1e-5)
        # Prediction with noise should make distribution broader (Z values closer to 0)
        self.assertTrue(all(B2.Z >= self.B.Z))

    def test_update_identity_at_mode(self):
        self.filter.filter_state = self.B
        self.filter.update_identity(self.Bnoise, self.B.mode())
        B3 = self.filter.filter_state
        self.assertIsInstance(B3, BinghamDistribution)
        # Mode should remain the same (up to antipodal sign)
        npt.assert_allclose(abs(self.B.mode()), abs(B3.mode()), atol=1e-5)
        # Update at mode should make distribution sharper (Z values more negative)
        self.assertTrue(all(B3.Z <= self.B.Z))

    def test_update_identity_different_measurement(self):
        self.filter.filter_state = self.B
        z = self.B.mode() + array([0.1, 0.0])
        z = z / linalg.norm(z)
        self.filter.update_identity(self.Bnoise, z)
        B4 = self.filter.filter_state
        self.assertIsInstance(B4, BinghamDistribution)
        self.assertTrue(all(B4.Z <= self.B.Z))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_nonlinear_identity_function(self):
        self.filter.filter_state = self.B
        self.filter.predict_nonlinear(lambda x: x, self.Bnoise)
        B5 = self.filter.filter_state
        self.assertIsInstance(B5, BinghamDistribution)
        npt.assert_allclose(abs(self.B.M), abs(B5.M), rtol=1e-5, atol=1e-5)
        self.assertTrue(all(B5.Z >= self.B.Z))


class TestBinghamFilter4D(unittest.TestCase):
    def setUp(self):
        Z = array([-5.0, -3.0, -2.0, 0.0])
        M = array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        self.B = BinghamDistribution(Z, M)
        self.filter = BinghamFilter()
        self.Bnoise = BinghamDistribution(
            array([-2.0, -2.0, -2.0, 0.0]),
            array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    def test_set_state_and_get_estimate(self):
        self.filter.filter_state = self.B
        B1 = self.filter.filter_state
        self.assertIsInstance(B1, BinghamDistribution)
        npt.assert_array_equal(self.B.M, B1.M)
        npt.assert_array_equal(self.B.Z, B1.Z)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_identity(self):
        self.filter.filter_state = self.B
        self.filter.predict_identity(self.Bnoise)
        B_pred = self.filter.filter_state
        self.assertIsInstance(B_pred, BinghamDistribution)
        npt.assert_allclose(self.B.M, B_pred.M, atol=1e-10)
        self.assertTrue(all(B_pred.Z >= self.B.Z))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_nonlinear_identity_function(self):
        self.filter.filter_state = self.B
        self.filter.predict_nonlinear(lambda x: x, self.Bnoise)
        B_nl = self.filter.filter_state
        self.assertIsInstance(B_nl, BinghamDistribution)
        npt.assert_allclose(abs(self.B.M), abs(B_nl.M), atol=1e-10)
        self.assertTrue(all(B_nl.Z >= self.B.Z))

        # predict_nonlinear with identity should approximate predict_identity
        self.filter.filter_state = self.B
        self.filter.predict_identity(self.Bnoise)
        B_id = self.filter.filter_state
        npt.assert_allclose(abs(B_id.M), abs(B_nl.M), atol=1e-10)
        npt.assert_allclose(B_id.Z, B_nl.Z, rtol=0.15)

    def test_update_identity_at_mode(self):
        self.filter.filter_state = self.B
        self.filter.update_identity(self.Bnoise, self.B.mode())
        B3 = self.filter.filter_state
        self.assertIsInstance(B3, BinghamDistribution)
        npt.assert_allclose(self.B.mode(), B3.mode(), atol=1e-10)
        self.assertTrue(all(B3.Z <= self.B.Z))

    def test_update_identity_different_measurement(self):
        self.filter.filter_state = self.B
        z = self.B.mode() + array([0.1, 0.1, 0.0, 0.0])
        z = z / linalg.norm(z)
        self.filter.update_identity(self.Bnoise, z)
        B4 = self.filter.filter_state
        self.assertIsInstance(B4, BinghamDistribution)
        self.assertTrue(all(B4.Z <= self.B.Z))


if __name__ == "__main__":
    unittest.main()
