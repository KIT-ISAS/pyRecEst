import numpy as np
from pyrecest.evaluation import generate_groundtruth, generate_measurements

def generate_simulated_scenarios(
    scenario_param,
    all_seeds,
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

    groundtruths = np.empty((np.size(all_seeds), scenario_param["n_steps"], 2))
    measurements = np.empty((np.size(all_seeds), scenario_param["n_steps"], 2))

    for run in range(np.size(all_seeds)):
        np.random.seed(run)
        groundtruths[run, :] = generate_groundtruth(scenario_param)
        measurements[run, :] = generate_measurements(
            groundtruths[run, :], scenario_param)
        
    return groundtruths, measurements