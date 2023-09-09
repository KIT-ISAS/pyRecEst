from typing import Any

import numpy as np
import random

from beartype import beartype
from .scenario_database import scenario_database
from .generate_simulated_scenarios import generate_simulated_scenarios


# pylint: disable=R0913,R0914
def start_evaluation(
    scenario: str | dict[str, Any],
    filter_configs: list[dict[str, Any]],
    n_runs: int,
    save_folder: str = ".",
    plot_each_step: bool = False,
    convert_to_point_estimate_during_runtime: bool = False,
    extract_all_point_estimates: bool = False,
    scenario_customization_params: None | dict = None,
    tolerate_failure: bool = False,
    initial_seed: None | int | np.uint32 = None,
    consecutive_seed: bool = False,
    auto_warning_on_off: bool = False,
) -> tuple[
    dict,
    list[dict],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray[np.ndarray],
]:
    """
    Main function for evaluating filters.

    Parameters:
        scenario (Union[str, Dict[str, Any]]): Scenario name or full parameterization.
        filters (List[Dict[str, Any]]): List of filters with names and parameters.
        no_runs (int): Number of runs.

    Optional Parameters:
        save_folder (str): The directory where results will be saved. Default is '.'.
        plot_each_step (bool): Whether to plot each step or not. Useful for debugging. Default is False.
        convert_to_point_estimate_during_runtime (bool): Whether to convert to point estimates during runtime or not. Default is False.
        extract_all_point_estimates (bool): Whether to extract all point estimates or not. Default is False.
        scenario_customization_params (Union[None, Dict]): Optional scenario parameters. Default is None.
        tolerate_failure (bool): Whether to tolerate failures or not. Default is False.
        initial_seed (Union[None, np.uint32]): The initial random seed. Default is a random 32-bit integer.
        consecutive_seed (bool): Whether to use consecutive seeds or not. Default is False.
        auto_warning_on_off (bool): Whether to automatically turn warnings on or off. Useful for debugging. Default is False.

    Returns:
        Tuple: results, groundtruths, and scenario parameters.
    """
    if isinstance(scenario, dict):
        scenario_param = scenario
        scenario_param["name"] = "custom"
    else:
        scenario_param = scenario_database(scenario, scenario_customization_params)
        scenario_param["name"] = scenario

    scenario_param["all_seeds"] = get_all_seeds(n_runs, initial_seed, consecutive_seed)

    start_evaluation(scenario_param, filter_configs, n_runs, save_folder,
                     plot_each_step, convert_to_point_estimate_during_runtime,
                     extract_all_point_estimates, scenario_customization_params,
                     tolerate_failure, auto_warning_on_off)

@beartype
def get_all_seeds(n_runs: int, seed_input=None, consecutive_seed:bool=True):
    if seed_input is None:
        seed_input = np.uint32(random.randint(1, 0xFFFFFFFF))  # nosec
        
    if np.size(seed_input) == n_runs:
        all_seeds = seed_input
    elif np.size(seed_input) == 1 and n_runs > 1:
        if consecutive_seed:
            all_seeds = list(range(seed_input, seed_input + n_runs))
        else:
            random.seed(seed_input)
            all_seeds = [
                random.randint(1, 0xFFFFFFFF) for _ in range(n_runs)  # nosec
            ]
    else:
        raise ValueError("The number of seeds provided must be either 1 or equal to the number of runs.")

    return all_seeds