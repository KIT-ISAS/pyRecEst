import unittest

import numpy as np
from pyrecest.evaluation import (
    check_and_fix_params,
    generate_groundtruth,
    generate_measurements,
    scenario_database,
)


class TestEvalation(unittest.TestCase):
    def test_generate_gt_R2(self):
        scenario_name = "R2randomWalk"
        scenario_param = scenario_database(scenario_name)
        scenario_param = check_and_fix_params(scenario_param)

        groundtruth = generate_groundtruth(np.zeros(2), scenario_param)

        # Test if groundtruth has the shape (timesteps, 2)
        self.assertEqual(groundtruth.shape, (scenario_param["timesteps"], 2))

    def test_generate_meas_R2(self):
        scenario_name = "R2randomWalk"
        scenario_param = scenario_database(scenario_name)
        scenario_param = check_and_fix_params(scenario_param)
        timesteps = 10

        measurements = generate_measurements(np.zeros((timesteps, 2)), scenario_param)

        self.assertEqual(len(measurements), timesteps)
        for i in range(timesteps):
            self.assertEqual(
                measurements[i].shape[0],
                scenario_param["n_meas_at_individual_time_step"][i],
            )


if __name__ == "__main__":
    unittest.main()
