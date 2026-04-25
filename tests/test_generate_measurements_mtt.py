import unittest

import numpy as np

from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.evaluation import check_and_fix_config, generate_measurements


class TestGenerateMeasurementsMtt(unittest.TestCase):
    def test_mtt_default_detection_probability_and_measurement_shape(self):
        simulation_param = check_and_fix_config(
            {
                "mtt": True,
                "eot": False,
                "n_timesteps": 2,
                "n_targets": 1,
                "initial_prior": GaussianDistribution(array([0.0, 0.0]), eye(2)),
                "meas_matrix_for_each_target": eye(2),
                "meas_noise": GaussianDistribution(
                    array([0.0, 0.0]), 1.0e-12 * eye(2)
                ),
            }
        )
        groundtruth = np.array(
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ]
        )

        measurements = generate_measurements(groundtruth, simulation_param)

        self.assertEqual(measurements[0].shape, (1, 2))
        self.assertEqual(measurements[1].shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
