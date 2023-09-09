import warnings

import numpy as np
from pyrecest.distributions import GaussianDistribution


def simulation_database(scenario, scenario_customization_params=None):
    simulation_param = {
        "initial_prior": lambda: "Scenario param not initialized",
        "n_timesteps": None,
        "all_seeds": None,
    }

    if scenario == "R2randomWalk":
        simulation_param["manifold"] = "Euclidean"
        simulation_param["n_timesteps"] = 10
        simulation_param["initial_prior"] = GaussianDistribution(
            np.zeros(2), 0.5 * np.eye(2)
        )
        simulation_param["meas_noise"] = GaussianDistribution(
            np.zeros(2), 0.5 * np.eye(2)
        )
        simulation_param["sys_noise"] = GaussianDistribution(np.zeros(2), 0.5 * np.eye(2))
        simulation_param["gen_next_state_without_noise_is_vectorized"] = True
    else:
        if scenario_customization_params is None:
            raise ValueError("Scenario not recognized.")

        warnings.warn(
            "Scenario not recognized. Assuming scenario_customization_params contains all parameters."
        )
        simulation_param = scenario_customization_params

        return simulation_param

    return simulation_param
