import unittest

import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import GoalConditionedReplayIMMFilter


class TestGoalConditionedReplayIMMFilter(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_predict_replay_moves_velocity_toward_goal(self):
        np.random.seed(0)

        filt = GoalConditionedReplayIMMFilter(
            initial_state=(np.array([0.0, 0.0]), 0.01 * np.eye(2)),
            candidate_goals=np.array([[1.0, 0.0]]),
            dt=1.0,
            attraction_strength=1.0,
            velocity_decay=0.5,
            smooth_sys_noise_cov=0.01 * np.eye(4),
            jump_sys_noise_cov=0.10 * np.eye(4),
            jump_probability=0.0,
        )

        filt.predict_replay()
        velocity_estimate = filt.get_velocity_estimate()

        self.assertEqual(velocity_estimate.shape, (2,))
        self.assertGreater(velocity_estimate[0], 0.2)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_position_update_returns_log_marginal_and_pulls_mean_to_measurement(self):
        np.random.seed(1)

        filt = GoalConditionedReplayIMMFilter(
            initial_state=(np.array([0.0, 0.0]), 0.02 * np.eye(2)),
            candidate_goals=np.array([[1.0, 0.0]]),
            dt=1.0,
            attraction_strength=1.0,
            velocity_decay=0.0,
            smooth_sys_noise_cov=0.01 * np.eye(4),
            jump_sys_noise_cov=0.10 * np.eye(4),
            jump_probability=0.0,
        )

        filt.predict_replay()
        log_marginal = filt.update_position(
            np.array([1.0, 0.0]),
            GaussianDistribution(np.zeros(2), 0.05 * np.eye(2)),
            return_log_marginal=True,
        )
        position_estimate = filt.get_position_estimate()

        self.assertTrue(np.isfinite(log_marginal))
        self.assertGreater(position_estimate[0], 0.55)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_goal_update_reweights_goal_posterior(self):
        np.random.seed(2)

        filt = GoalConditionedReplayIMMFilter(
            initial_state=(np.array([0.0, 0.0]), 0.01 * np.eye(2)),
            candidate_goals=np.array([[-1.0, 0.0], [1.0, 0.0]]),
            goal_prior=np.array([0.5, 0.5]),
            dt=1.0,
            smooth_sys_noise_cov=0.01 * np.eye(4),
            jump_sys_noise_cov=0.10 * np.eye(4),
            jump_probability=0.0,
        )

        log_marginal = filt.update_goal(
            np.array([1.0, 0.0]),
            GaussianDistribution(np.zeros(2), 0.01 * np.eye(2)),
            return_log_marginal=True,
        )
        goal_probabilities = filt.goal_probabilities

        self.assertTrue(np.isfinite(log_marginal))
        self.assertGreater(goal_probabilities[1], 0.95)
        self.assertGreater(goal_probabilities[1], goal_probabilities[0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_linear_association_likelihood_is_positive(self):
        np.random.seed(3)

        filt = GoalConditionedReplayIMMFilter(
            initial_state=(np.array([0.5, 0.0, 0.2, 0.0]), 0.05 * np.eye(4)),
            candidate_goals=np.array([[1.0, 0.0]]),
            dt=1.0,
            smooth_sys_noise_cov=0.01 * np.eye(4),
            jump_sys_noise_cov=0.10 * np.eye(4),
            jump_probability=0.0,
        )

        H_vel = np.zeros((2, 4))
        H_vel[:, 2:4] = np.eye(2)
        meas_noise = GaussianDistribution(np.zeros(2), 0.05 * np.eye(2))

        assoc = filt.association_likelihood_linear(
            np.array([0.2, 0.0]),
            H_vel,
            meas_noise,
        )

        self.assertGreater(assoc, 0.0)
