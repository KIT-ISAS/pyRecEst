import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, isinf, isnan, logical_or, random, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import GoalConditionedReplayParticleFilter

from .test_goal_conditioned_replay_common import (
    assert_association_likelihood_linear_positive,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported"
)
class TestGoalConditionedReplayParticleFilter(unittest.TestCase):
    def test_predict_replay_moves_velocity_toward_goal(self):
        random.seed(0)

        filt = GoalConditionedReplayParticleFilter(
            n_particles=256,
            spatial_dim=2,
            dt=1.0,
            alpha=0.5,
            beta=1.0,
            process_noise=GaussianDistribution(zeros(2), 0.01 * eye(2)),
            goal_noise=GaussianDistribution(zeros(2), 0.001 * eye(2)),
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.0, 0.0]),
            velocity_prior=array([0.0, 0.0]),
            goal_prior=array([1.0, 0.0]),
        )

        filt.predict_replay()
        velocity_estimate = filt.get_velocity_estimate()

        self.assertEqual(velocity_estimate.shape, (2,))
        self.assertGreater(float(velocity_estimate[0]), 0.2)

    def test_position_update_returns_log_marginal_and_pulls_particles_to_measurement(
        self,
    ):
        random.seed(1)

        filt = GoalConditionedReplayParticleFilter(
            n_particles=512,
            spatial_dim=2,
            dt=1.0,
            alpha=0.0,
            beta=1.0,
            process_noise=GaussianDistribution(zeros(2), 0.02 * eye(2)),
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.0, 0.0]),
            velocity_prior=array([0.0, 0.0]),
            goal_prior=array([1.0, 0.0]),
        )
        filt.predict_replay(use_semi_implicit_position_update=True)

        meas_noise = GaussianDistribution(zeros(2), 0.05 * eye(2))
        log_marginal = filt.update_position(
            meas_noise,
            array([1.0, 0.0]),
            return_log_marginal=True,
        )
        position_estimate = filt.get_position_estimate()

        self.assertFalse(logical_or(isnan(log_marginal), isinf(log_marginal)))
        self.assertGreater(float(position_estimate[0]), 0.5)

    def test_linear_association_likelihood_is_positive(self):
        random.seed(2)

        filt = GoalConditionedReplayParticleFilter(
            n_particles=128,
            spatial_dim=2,
            process_noise=GaussianDistribution(zeros(2), 0.01 * eye(2)),
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.5, 0.0]),
            velocity_prior=array([0.2, 0.0]),
            goal_prior=array([1.0, 0.0]),
        )

        assert_association_likelihood_linear_positive(self, filt, state_dim=6)

    def test_position_likelihood_with_proposal_rejuvenates_particles(self):
        random.seed(3)

        filt = GoalConditionedReplayParticleFilter(
            n_particles=256,
            spatial_dim=2,
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.0, 0.0]),
            velocity_prior=array([0.0, 0.0]),
            goal_prior=array([1.0, 0.0]),
        )
        proposal_positions = array(
            [
                [0.8, 0.0],
                [1.0, 0.0],
                [1.2, 0.0],
            ]
        )

        def likelihood(positions):
            residual = positions - array([1.0, 0.0])
            return 1.0 / (
                1.0 + residual[:, 0] * residual[:, 0] + residual[:, 1] * residual[:, 1]
            )

        log_marginal = filt.update_position_likelihood_with_proposal(
            likelihood,
            position_proposal=proposal_positions,
            proposal_weights=array([0.2, 0.6, 0.2]),
            proposal_probability=1.0,
            return_log_marginal=True,
        )
        position_estimate = filt.get_position_estimate()

        self.assertFalse(logical_or(isnan(log_marginal), isinf(log_marginal)))
        self.assertGreater(float(position_estimate[0]), 0.9)
        self.assertGreater(float(filt.last_position_proposal_fraction), 0.99)


if __name__ == "__main__":
    unittest.main()
