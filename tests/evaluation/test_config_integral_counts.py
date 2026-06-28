import pytest

from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation.check_and_fix_config import (
    _expand_meas_per_step,
    _validate_measurement_counts,
    check_and_fix_config,
)


def _base_config(**overrides):
    config = {
        "n_timesteps": 3,
        "initial_prior": GaussianDistribution(zeros(1), eye(1)),
    }
    config.update(overrides)
    return config


def test_expand_meas_per_step_rejects_true_as_count():
    with pytest.raises(TypeError, match="integer"):
        _expand_meas_per_step({"n_timesteps": 4, "meas_per_step": True})


def test_validate_measurement_counts_rejects_true_entries():
    with pytest.raises(TypeError, match="integer"):
        _validate_measurement_counts(
            {
                "n_timesteps": 2,
                "n_meas_at_individual_time_step": [1, True],
            }
        )


def test_check_and_fix_config_rejects_true_as_timestep_count():
    with pytest.raises(TypeError, match="n_timesteps"):
        check_and_fix_config(_base_config(n_timesteps=True))
