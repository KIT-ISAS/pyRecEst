import unittest
import numpy as np

from pyrecest.evaluation import generate_groundtruth, generate_measurements, scenario_database, check_and_fix_params


class TestEvalation(unittest.TestCase):
    noRunsDefault = 10  # Equivalent of MATLAB's Constant property

    def test_generate_gt_R2(self):
        # Your setup code here, e.g.,
        scenarioName = 'R2randomWalk'
        scenarioParam = scenario_database(scenarioName)
        scenarioParam = check_and_fix_params(scenarioParam)
        
        groundtruth = generate_groundtruth(scenarioParam)
        
        
    def test_generate_gt_R2(self):
        # Your setup code here, e.g.,
        scenarioName = 'R2randomWalk'
        scenarioParam = scenario_database(scenarioName)
        scenarioParam = check_and_fix_params(scenarioParam)
        
        measurements = generate_measurements(np.zeros((2, 10)), scenarioParam)

if __name__ == '__main__':
    unittest.main()
