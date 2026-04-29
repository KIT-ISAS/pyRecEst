import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member,duplicate-code
from pyrecest.backend import array, concatenate, eye, isinf, isnan, ones, random, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import GoalConditionedReplayParticleIMMFilter

from .test_goal_conditioned_replay_common import (
    assert_association_likelihood_linear_positive,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax", reason="Backend not supported"
)
class TestGoalConditionedReplayParticleIMMFilter(unittest.TestCase):
    def test_goal_directed_mode_moves_velocity_toward_goal(self):
        random.seed(0)
        goal_mode = GoalConditionedReplayParticleIMMFilter.mode_names.index(
            "goal_directed"
        )
        mode_prior = zeros((len(GoalConditionedReplayParticleIMMFilter.mode_names),))
        mode_prior[goal_mode] = 1.0

        filt = GoalConditionedReplayParticleIMMFilter(
            n_particles=256,
            spatial_dim=2,
            dt=1.0,
            alpha=0.5,
            beta=1.0,
            process_noise=GaussianDistribution(zeros(2), 0.01 * eye(2)),
            mode_prior=mode_prior,
            mode_stickiness=1.0,
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.0, 0.0]),
            velocity_prior=array([0.0, 0.0]),
            goal_prior=array([1.0, 0.0]),
        )

        filt.predict_replay()
        velocity_estimate = filt.get_velocity_estimate()

        self.assertEqual(velocity_estimate.shape, (2,))
        self.assertEqual(filt.most_likely_mode(), "goal_directed")
        self.assertGreater(float(velocity_estimate[0]), 0.2)

    def test_mode_transition_can_force_jump_toward_goal(self):
        random.seed(1)
        names = GoalConditionedReplayParticleIMMFilter.mode_names
        stationary_mode = names.index("stationary")
        jump_mode = names.index("jump")
        mode_prior = zeros((len(names),))
        mode_prior[stationary_mode] = 1.0
        transition_matrix = eye(len(names))
        transition_matrix[stationary_mode] = zeros((len(names),))
        transition_matrix[stationary_mode, jump_mode] = 1.0

        filt = GoalConditionedReplayParticleIMMFilter(
            n_particles=64,
            spatial_dim=2,
            dt=1.0,
            mode_prior=mode_prior,
            mode_transition_matrix=transition_matrix,
            jump_fraction=0.8,
        )
        filt.initialize_from_state_priors(
            position_prior=array([0.0, 0.0]),
            velocity_prior=array([0.0, 0.0]),
            goal_prior=array([10.0, 0.0]),
        )

        filt.predict_replay()
        position_estimate = filt.get_position_estimate()

        self.assertEqual(filt.most_likely_mode(), "jump")
        self.assertGreater(float(filt.last_mode_transition_fraction), 0.99)
        self.assertGreater(float(position_estimate[0]), 7.5)

    def test_position_update_resamples_mode_indices_with_particles(self):
        random.seed(2)
        names = GoalConditionedReplayParticleIMMFilter.mode_names
        stationary_mode = names.index("stationary")
        goal_mode = names.index("goal_directed")
        n_particles = 400
        half = n_particles // 2

        filt = GoalConditionedReplayParticleIMMFilter(
            n_particles=n_particles,
            spatial_dim=2,
            mode_stickiness=1.0,
        )
        near_origin = zeros((half, 2))
        near_measurement = concatenate([ones((half, 1)), zeros((half, 1))], axis=1)
        positions = concatenate([near_origin, near_measurement], axis=0)
        modes = concatenate(
            [
                ones((half,)) * stationary_mode,
                ones((half,)) * goal_mode,
            ],
            axis=0,
        )
        filt.set_state_components(
            positions=positions,
            velocities=zeros((n_particles, 2)),
            goals=positions,
        )
        filt.set_mode_indices(modes)

        log_marginal = filt.update_position(
            array([1.0, 0.0]),
            GaussianDistribution(zeros(2), 0.005 * eye(2)),
            return_log_marginal=True,
        )

        self.assertFalse(isnan(log_marginal))
        self.assertFalse(isinf(log_marginal))
        self.assertGreater(float(filt.mode_probabilities[goal_mode]), 0.9)

    def test_linear_association_likelihood_is_positive(self):
        random.seed(3)

        filt = GoalConditionedReplayParticleIMMFilter(
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


if __name__ == "__main__":
    unittest.main()
