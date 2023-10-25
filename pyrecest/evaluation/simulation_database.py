import warnings
from typing import Optional

from beartype import beartype

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import eye, zeros
from pyrecest.distributions import GaussianDistribution


@beartype
def simulation_database(
    scenario_name: str = "custom", scenario_customization_params: Optional[dict] = None
) -> dict:
    simulation_param: dict = {
        "n_timesteps": None,
        "all_seeds": None,
        "eot": False,
        "mtt": False,
        "n_targets": 1,
    }

    if scenario_name == "custom":
        warnings.warn(
            "Scenario not recognized. Assuming scenario_customization_params contains all parameters."
        )
        assert scenario_customization_params is not None
        simulation_param.update(scenario_customization_params)
    elif scenario_name == "R2randomWalk":
        simulation_param["manifold"] = "Euclidean"
        simulation_param["n_timesteps"] = 10
        simulation_param["initial_prior"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["meas_noise"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["sys_noise"] = GaussianDistribution(zeros(2), 0.5 * eye(2))
        simulation_param["gen_next_state_without_noise_is_vectorized"] = True
    else:
        raise ValueError("Scenario not recognized.")

    return simulation_param
