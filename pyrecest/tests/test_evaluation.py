import unittest

import numpy as np
from parameterized import parameterized
from pyrecest.distributions import (
    GaussianDistribution,
    HypertoroidalWrappedNormalDistribution,
)
from pyrecest.evaluation import (
    check_and_fix_params,
    configure_for_filter,
    determine_all_deviations,
    generate_groundtruth,
    generate_measurements,
    iterate_configs_and_runs,
    perform_predict_update_cycles,
    scenario_database,
)
from pyrecest.filters import HypertoroidalParticleFilter, KalmanFilter


class TestEvalation(unittest.TestCase):
    def setUp(self):
        self.scenario_name = "R2randomWalk"
        scenario_param = scenario_database(self.scenario_name)
        self.scenario_param = check_and_fix_params(scenario_param)
        self.timesteps = 10

    @parameterized.expand(
        [
            (np.zeros(2),),
            (None,),
        ]
    )
    def test_generate_gt_R2(self, x0):
        groundtruth = generate_groundtruth(self.scenario_param, x0)

        # Test if groundtruth has the shape (timesteps, 2)
        self.assertEqual(groundtruth.shape, (self.timesteps, 2))

    def test_generate_meas_R2(self):
        measurements = generate_measurements(
            np.zeros((self.timesteps, 2)), self.scenario_param
        )

        self.assertEqual(len(measurements), self.timesteps)
        for i in range(self.timesteps):
            self.assertEqual(
                measurements[i].shape[0],
                self.scenario_param["n_meas_at_individual_time_step"][i],
            )

    def test_determine_all_deviations(self):
        def dummy_extract_mean(x):
            return x
        
        def dummy_distance_function(x, y):
            return np.linalg.norm(x - y)

        groundtruths = np.array([[[1, 2, 3], [2, 3, 4]], [[11, 12, 13], [12, 13, 14]]])
        results = [
            {
                "filter_name": "filter1",
                "filter_param": "params1",
                "last_estimates": groundtruths[:, -1, :],
            },
            {
                "filter_name": "filter2",
                "filter_param": "params2",
                "last_estimates": groundtruths[:, -1, :] + 1,
            },
        ]

        # Run the function and get the deviations matrix
        all_deviations = determine_all_deviations(
            results,
            dummy_extract_mean,
            dummy_distance_function,
            groundtruths,
        )

        # Check the shape of the output matrices
        assert len(all_deviations) == len(results)

        # Validate some of the results
        np.testing.assert_allclose(
            # Should be zeros as the lastEstimates match groundtruths
            all_deviations[0], [0, 0]
        )
        np.testing.assert_allclose(
            # Should be np.sqrt(2) away from groundtruths
            all_deviations[1], [np.sqrt(3), np.sqrt(3)]
        )

    def test_configure_kf(self):
        filterParam = {"name": "kf", "parameter": None}
        scenarioParam = {
            "initial_prior": GaussianDistribution(np.array([0, 0]), np.eye(2)),
            "inputs": None,
            "manifold_type": "Euclidean",
            "meas_noise": GaussianDistribution(np.array([0, 0]), np.eye(2)),
        }

        (
            configured_filter,
            predictionRoutine,
            _,
            meas_noise_for_filter,
        ) = configure_for_filter(filterParam, scenarioParam)

        self.assertIsInstance(configured_filter, KalmanFilter)
        self.assertIsNotNone(predictionRoutine)
        self.assertIsInstance(meas_noise_for_filter, np.ndarray)

    def test_configure_pf(self):
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

    def test_configure_unsupported_filter(self):
        filterParam = {"name": "unsupported_filter", "parameter": 10}
        scenario_param = {
            "initial_prior": "some_initial_prior",
            "inputs": None,
            "manifold_type": "Euclidean",
        }

        with self.assertRaises(ValueError):
            configure_for_filter(filterParam, scenario_param)

    def test_perform_predict_update_cycles(self):
        scenario_name = "R2randomWalk"
        scenario_param = scenario_database(scenario_name)
        scenario_param = check_and_fix_params(scenario_param)
        timesteps = 10

        (
            time_elapsed,
            last_filter_state,
            last_estimate,
            all_estimates,
        ) = perform_predict_update_cycles(
            scenario_param,
            {"name": "kf", "parameter": None},
            np.zeros((timesteps, 2)),
            generate_measurements(np.zeros((timesteps, 2)), scenario_param),
        )

        self.assertIsInstance(time_elapsed, float)
        self.assertGreater(time_elapsed, 0)
        self.assertIsNotNone(last_filter_state)
        self.assertIsInstance(last_estimate, np.ndarray)
        self.assertEqual(last_estimate.shape, (2,))
        self.assertIsNone(all_estimates)

    def test_iterate_configs_and_runs_kf_only(self):
        scenario_name = "R2randomWalk"
        scenario_param = scenario_database(scenario_name)
        scenario_param = check_and_fix_params(scenario_param)
        scenario_param["timesteps"] = 10

        iterate_configs_and_runs(
            scenario_param, [{"name": "kf", "filter_params": None}], n_runs=10
        )

    def test_iterate_configs_and_runs_kf_and_pf(self):
        scenario_name = "R2randomWalk"
        scenario_param = scenario_database(scenario_name)
        scenario_param = check_and_fix_params(scenario_param)
        scenario_param["timesteps"] = 10

        iterate_configs_and_runs(
            scenario_param,
            [
                {"name": "kf", "filter_params": None},
                {"name": "pf", "filter_params": 100},
            ],
            n_runs=10,
        )


if __name__ == "__main__":
    unittest.main()
