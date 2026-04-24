import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import random

from .check_and_fix_config import check_and_fix_config
from .generate_groundtruth import generate_groundtruth
from .generate_measurements import generate_measurements


def generate_simulated_scenarios(
    simulation_params,
):
    """
    Generate simulated scenarios.

    Returns
    -------
    groundtruths : numpy.ndarray
        The groundtruths.
    measurements : numpy.ndarray
        The measurements.

    """
    simulation_params = check_and_fix_config(simulation_params)
    all_seeds = simulation_params["all_seeds"]
    try:
        all_seeds = list(all_seeds)
    except TypeError:
        all_seeds = [all_seeds]
    n_runs = len(all_seeds)

    groundtruths = np.empty(
        (n_runs, simulation_params["n_timesteps"]),
        dtype=object,
    )
    measurements = np.empty(
        (n_runs, simulation_params["n_timesteps"]),
        dtype=object,
    )

    for run, seed in enumerate(all_seeds):
        random.seed(seed)
        groundtruths[run, :] = generate_groundtruth(simulation_params)
        measurements[run, :] = generate_measurements(
            groundtruths[run, :], simulation_params
        )

    return groundtruths, measurements
