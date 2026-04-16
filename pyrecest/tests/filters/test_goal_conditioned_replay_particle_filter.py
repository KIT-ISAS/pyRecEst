import numpy as np

from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import GoalConditionedReplayParticleFilter


def test_predict_replay_moves_velocity_toward_goal():
    np.random.seed(0)

    filt = GoalConditionedReplayParticleFilter(
        n_particles=256,
        spatial_dim=2,
        dt=1.0,
        alpha=0.5,
        beta=1.0,
        process_noise=GaussianDistribution(np.zeros(2), 0.01 * np.eye(2)),
        goal_noise=GaussianDistribution(np.zeros(2), 0.001 * np.eye(2)),
    )
    filt.initialize_from_state_priors(
        position_prior=np.array([0.0, 0.0]),
        velocity_prior=np.array([0.0, 0.0]),
        goal_prior=np.array([1.0, 0.0]),
    )

    filt.predict_replay()
    velocity_estimate = filt.get_velocity_estimate()

    assert velocity_estimate.shape == (2,)
    assert velocity_estimate[0] > 0.2


def test_position_update_returns_log_marginal_and_pulls_particles_to_measurement():
    np.random.seed(1)

    filt = GoalConditionedReplayParticleFilter(
        n_particles=512,
        spatial_dim=2,
        dt=1.0,
        alpha=0.0,
        beta=1.0,
        process_noise=GaussianDistribution(np.zeros(2), 0.02 * np.eye(2)),
    )
    filt.initialize_from_state_priors(
        position_prior=np.array([0.0, 0.0]),
        velocity_prior=np.array([0.0, 0.0]),
        goal_prior=np.array([1.0, 0.0]),
    )
    filt.predict_replay(use_semi_implicit_position_update=True)

    meas_noise = GaussianDistribution(np.zeros(2), 0.05 * np.eye(2))
    log_marginal = filt.update_position(
        meas_noise,
        np.array([1.0, 0.0]),
        return_log_marginal=True,
    )
    position_estimate = filt.get_position_estimate()

    assert np.isfinite(log_marginal)
    assert position_estimate[0] > 0.5


def test_linear_association_likelihood_is_positive():
    np.random.seed(2)

    filt = GoalConditionedReplayParticleFilter(
        n_particles=128,
        spatial_dim=2,
        process_noise=GaussianDistribution(np.zeros(2), 0.01 * np.eye(2)),
    )
    filt.initialize_from_state_priors(
        position_prior=np.array([0.5, 0.0]),
        velocity_prior=np.array([0.2, 0.0]),
        goal_prior=np.array([1.0, 0.0]),
    )

    H_vel = np.zeros((2, 6))
    H_vel[:, 2:4] = np.eye(2)
    meas_noise = GaussianDistribution(np.zeros(2), 0.05 * np.eye(2))

    assoc = filt.association_likelihood_linear(
        np.array([0.2, 0.0]),
        H_vel,
        meas_noise,
    )

    assert assoc > 0.0
