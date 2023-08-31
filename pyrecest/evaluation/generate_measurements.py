import numpy as np


def generate_measurements(groundtruth, scenario_param):
    """
    Generate measurements based on the given groundtruth and scenario parameters.
    
    Parameters:
        groundtruth (ndarray): Ground truth data.
        scenario_param (dict): Dictionary containing scenario parameters.
        
    Returns:
        measurements (list): List of generated measurements at each time step.
    """
    
    measurements = np.empty(scenario_param['timesteps'])
    
    if 'MTT' in scenario_param.get('manifoldType', ''):
        # No support for clutter rate and multiple measurements currently
        assert scenario_param['clutterRate'] == 0, "Clutter currently not supported."
        
        n_observations = np.random.binomial(1, scenario_param['detectionProbability'], (scenario_param['timesteps'], scenario_param['nTargets']))
        
        for t in range(scenario_param['timesteps']):
            n_meas_at_t = np.sum(n_observations[t, :])
            measurements[t] = np.nan * np.zeros((scenario_param['measMatrixForEachTarget'].shape[0], n_meas_at_t))
            
            meas_no = 0
            for target_no in range(scenario_param['nTargets']):
                if n_observations[t, target_no] == 1:
                    meas_no += 1
                    measurements[t][:, meas_no-1] = np.dot(scenario_param['measMatrixForEachTarget'], groundtruth[:, t, target_no]) \
                        + scenario_param['measNoise'].sample(1)
                else:
                    assert n_observations[t, target_no] == 0, "Multiple measurements currently not supported."
            
            assert meas_no == n_meas_at_t, "Mismatch in number of measurements."
            
    else:
        raise NotImplementedError("Not yet impelmeneted.")
                        
    return measurements