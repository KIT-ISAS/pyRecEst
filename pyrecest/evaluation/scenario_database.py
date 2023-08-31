import numpy as np
from pyrecest.distributions import GaussianDistribution

def scenario_database(scenario, scenario_customization_params=None):
    scenario_param = {
        'initialPrior': lambda: "Scenario param not initialized",
        'timesteps': np.nan,
        'allSeeds': np.nan
    }
    
    if scenario == 'R2randomWalk':
        scenario_param['manifoldType'] = 'Euclidean'
        scenario_param['timesteps'] = 10
        scenario_param['initialPrior'] = GaussianDistribution([0, 0], 0.5 * np.eye(2))
        scenario_param['measNoise'] = GaussianDistribution([0, 0], 0.5 * np.eye(2))
        scenario_param['sysNoise'] = GaussianDistribution([0, 0], 0.5 * np.eye(2))
        scenario_param['genNextStateWithoutNoiseIsVectorized'] = True
    else:
        raise ValueError("Scenario not supported.")
    
    return scenario_param