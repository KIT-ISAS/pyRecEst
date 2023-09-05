import unittest

import numpy as np
from pyrecest.distributions import (
    GaussianDistribution,
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.evaluation import (
    check_and_fix_params,
    configure_for_filter,
    generate_groundtruth,
    generate_measurements,
    scenario_database,
)
from pyrecest.filters import HypertoroidalParticleFilter, KalmanFilter


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

    def test_kf_filter(self):
        filterParam = {"name": "kf", "parameter": 10}
        scenarioParam = {
            "initial_prior": GaussianDistribution(np.array([0, 0]), np.eye(2)),
            "inputs": None,
            "manifoldType": "Euclidean",
        }

        configured_filter, predictionRoutine, *_ = configure_for_filter(
            filterParam, scenarioParam
        )

        self.assertIsInstance(configured_filter, KalmanFilter)
        self.assertIsNotNone(predictionRoutine)

    def test_pf_filter(self):
        filterParam = {"name": "pf", "parameter": 100}
        scenarioParam = {
            "initial_prior": HypertoroidalWrappedNormalDistribution(
                np.array([0, 0]), np.eye(2)
            ),
            "inputs": None,
            "manifold_type": "hypertorus",
            "gen_next_state_with_noise": lambda x: x,
        }

        configured_filter, predictionRoutine, *_ = configure_for_filter(
            filterParam, scenarioParam
        )

        self.assertIsInstance(configured_filter, HypertoroidalParticleFilter)
        self.assertIsNotNone(predictionRoutine)

    def test_unsupported_filter(self):
        filterParam = {"name": "unsupported_filter", "parameter": 10}
        scenarioParam = {
            "initial_prior": "some_initial_prior",
            "inputs": None,
            "manifold_type": "Euclidean",
        }

        with self.assertRaises(ValueError):
            configure_for_filter(filterParam, scenarioParam)


if __name__ == "__main__":
    unittest.main()
