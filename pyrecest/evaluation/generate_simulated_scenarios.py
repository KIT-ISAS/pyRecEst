import numpy as np
from pyrecest.evaluation import generate_groundtruth, generate_measurements, check_and_fix_params


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
    simulation_params = check_and_fix_params(simulation_params)

    groundtruths = np.empty((np.size(simulation_params["all_seeds"]), simulation_params["n_timesteps"]), dtype=np.ndarray)
    measurements = np.empty((np.size(simulation_params["all_seeds"]), simulation_params["n_timesteps"]), dtype=np.ndarray)

    for run, seed in enumerate(simulation_params["all_seeds"]):
        np.random.seed(seed)
        groundtruths[run, :] = generate_groundtruth(simulation_params)
        measurements[run, :] = generate_measurements(
            groundtruths[run, :], simulation_params)
        
    return groundtruths, measurements