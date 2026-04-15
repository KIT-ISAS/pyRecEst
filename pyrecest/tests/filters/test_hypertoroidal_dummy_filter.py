import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.filters.hypertoroidal_dummy_filter import HypertoroidalDummyFilter


class HypertoroidalDummyFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter_t1 = HypertoroidalDummyFilter(1)
        self.filter_t2 = HypertoroidalDummyFilter(2)

    def test_dim_t1(self):
        self.assertEqual(self.filter_t1.dim, 1)

    def test_dim_t2(self):
        self.assertEqual(self.filter_t2.dim, 2)

    def test_assert_dim_too_small(self):
        with self.assertRaises(AssertionError):
            HypertoroidalDummyFilter(0)

    def test_filter_state_is_uniform(self):
        self.assertIsInstance(
            self.filter_t2.filter_state, HypertoroidalUniformDistribution
        )

    def test_get_point_estimate_shape_t1(self):
        est = self.filter_t1.get_point_estimate()
        self.assertEqual(est.shape, (1,))

    def test_get_point_estimate_shape_t2(self):
        est = self.filter_t2.get_point_estimate()
        self.assertEqual(est.shape, (2,))

    def test_predict_identity_is_noop(self):
        import numpy as np

        noise = HypertoroidalWrappedNormalDistribution(array([0.0, 0.0]), np.eye(2) * 0.5)
        state_before = self.filter_t2.filter_state
        self.filter_t2.predict_identity(noise)
        self.assertIs(self.filter_t2.filter_state, state_before)

    def test_predict_nonlinear_is_noop(self):
        state_before = self.filter_t2.filter_state
        self.filter_t2.predict_nonlinear(lambda x: x)
        self.assertIs(self.filter_t2.filter_state, state_before)

    def test_update_identity_is_noop(self):
        import numpy as np

        noise = HypertoroidalWrappedNormalDistribution(array([0.0, 0.0]), np.eye(2) * 0.5)
        measurement = array([1.0, 2.0])
        state_before = self.filter_t2.filter_state
        self.filter_t2.update_identity(noise, measurement)
        self.assertIs(self.filter_t2.filter_state, state_before)

    def test_update_nonlinear_is_noop(self):
        state_before = self.filter_t2.filter_state
        self.filter_t2.update_nonlinear(lambda z, x: x)
        self.assertIs(self.filter_t2.filter_state, state_before)

    def test_filter_state_setter_is_noop(self):
        original_state = self.filter_t2.filter_state
        new_dist = HypertoroidalUniformDistribution(2)
        self.filter_t2.filter_state = new_dist
        self.assertIs(self.filter_t2.filter_state, original_state)

    def test_set_state_is_noop(self):
        original_state = self.filter_t2.filter_state
        new_dist = HypertoroidalUniformDistribution(2)
        self.filter_t2.set_state(new_dist)
        self.assertIs(self.filter_t2.filter_state, original_state)

    def test_get_estimate_returns_distribution(self):
        est = self.filter_t2.get_estimate()
        self.assertIsInstance(est, HypertoroidalUniformDistribution)


if __name__ == "__main__":
    unittest.main()
