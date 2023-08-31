import numpy as np


def generate_groundtruth(x0, scenario_param):
    """
    Generate ground truth based on the given scenario parameters.
    
    Parameters:
        x0 (ndarray): Starting point.
        scenario_param (dict): Dictionary containing scenario parameters.
        
    Returns:
        groundtruth (ndarray): Generated ground truth.
    """
    
    assert x0.shape[2] == scenario_param['nTargets'], "Mismatch in number of targets."
    
    # Initialize ground truth
    groundtruth = np.fill((x0.shape[0], scenario_param['timesteps'], scenario_param['nTargets']), np.nan)
    
    for target_no in range(scenario_param['nTargets']):
        groundtruth[:, 0, target_no] = x0[:, :, target_no]
        
        if 'inputs' in scenario_param:
            assert scenario_param['inputs'].shape[1] == scenario_param['timesteps'] - 1, "Mismatch in number of timesteps."
        
        if 'genNextStateWithNoise' in scenario_param:
            for t in range(1, scenario_param['timesteps']):
                if 'inputs' not in scenario_param or scenario_param['inputs'] is None:
                    groundtruth[:, t, target_no] = scenario_param['genNextStateWithNoise'](groundtruth[:, t-1, target_no])
                else:
                    groundtruth[:, t, target_no] = scenario_param['genNextStateWithNoise'](groundtruth[:, t-1, target_no], scenario_param['inputs'][:, t-1])
        
        elif 'sysNoise' in scenario_param:
            for t in range(1, scenario_param['timesteps']):
                if 'genNextStateWithoutNoise' in scenario_param:
                    if 'inputs' not in scenario_param or scenario_param['inputs'] is None:
                        state_to_add_noise_to = scenario_param['genNextStateWithoutNoise'](groundtruth[:, t-1, target_no])
                    else:
                        state_to_add_noise_to = scenario_param['genNextStateWithoutNoise'](groundtruth[:, t-1, target_no], scenario_param['inputs'][:, t-1])
                else:
                    assert 'inputs' not in scenario_param or scenario_param['inputs'] is None, "No inputs accepted for the identity system model."
                    state_to_add_noise_to = groundtruth[:, t-1, target_no]
                
                groundtruth[:, t, target_no] = state_to_add_noise_to + scenario_param['sysNoise'].sample(1)
        
        else:
            raise ValueError("Cannot generate groundtruth.")
            
    return groundtruth