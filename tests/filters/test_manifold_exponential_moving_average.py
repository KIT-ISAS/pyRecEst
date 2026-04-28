import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.filters import ManifoldExponentialMovingAverage


def _phi_euclidean(state, xi):
    return state + xi


def _phi_inv_euclidean(state_ref, state):
    return state - state_ref


def _phi_so2(state, xi):
    return state + xi[0]


def _phi_inv_so2(state_ref, state):
    diff = state - state_ref
    return array([(diff + pi) % (2.0 * pi) - pi])


class TestManifoldExponentialMovingAverage(unittest.TestCase):
    def test_update_matches_euclidean_ema(self):
        ema = ManifoldExponentialMovingAverage(
            initial_state=array([0.0, 0.0]),
            alpha=0.25,
            phi=_phi_euclidean,
            phi_inv=_phi_inv_euclidean,
        )

        ema.update(array([4.0, -2.0]))
        npt.assert_allclose(ema.get_point_estimate(), array([1.0, -0.5]))

        ema.update(array([4.0, -2.0]))
        npt.assert_allclose(ema.filter_state, array([1.75, -0.875]))

    def test_first_update_initializes_empty_estimate(self):
        sample = array([2.0, 3.0])
        ema = ManifoldExponentialMovingAverage(
            initial_state=None,
            alpha=0.5,
            phi=_phi_euclidean,
            phi_inv=_phi_inv_euclidean,
        )

        ema.update(sample)

        npt.assert_allclose(ema.get_point_estimate(), sample)

    def test_circle_update_uses_shortest_tangent_direction(self):
        ema = ManifoldExponentialMovingAverage(
            initial_state=3.0,
            alpha=0.5,
            phi=_phi_so2,
            phi_inv=_phi_inv_so2,
        )

        ema.update(-3.0)

        npt.assert_allclose(ema.get_point_estimate(), pi, atol=1e-12)

    def test_alpha_must_be_between_zero_and_one(self):
        with self.assertRaises(ValueError):
            ManifoldExponentialMovingAverage(
                initial_state=0.0,
                alpha=1.1,
                phi=_phi_so2,
                phi_inv=_phi_inv_so2,
            )

        ema = ManifoldExponentialMovingAverage(
            initial_state=0.0,
            alpha=1.0,
            phi=_phi_so2,
            phi_inv=_phi_inv_so2,
        )
        with self.assertRaises(ValueError):
            ema.alpha = -0.1


if __name__ == "__main__":
    unittest.main()
