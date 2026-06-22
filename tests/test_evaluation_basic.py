import io
import os
import tempfile
import unittest
import warnings
from typing import Optional

import matplotlib
import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from parameterized import parameterized

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import all, array, eye, random, sqrt, zeros
from pyrecest.distributions import (
    GaussianDistribution,
    HypertoroidalWrappedNormalDistribution,
    VonMisesFisherDistribution,
)
from pyrecest.evaluation import (
    check_and_fix_config,
    configure_for_filter,
    determine_all_deviations,
    evaluate_for_file,
    evaluate_for_simulation_config,
    generate_groundtruth,
    generate_measurements,
    generate_simulated_scenarios,
    get_axis_label,
    get_distance_function,
    get_extract_mean,
    iterate_configs_and_runs,
    perform_predict_update_cycles,
    simulation_database,
    summarize_filter_results,
)
from pyrecest.filters import HypertoroidalParticleFilter, KalmanFilter
from shapely.geometry import Polygon


class TestEvalationBase(unittest.TestCase):
    scenario_name: Optional[str] = None

    def setUp(self):
        assert (
            self.scenario_name is not None
        ), "scenario_name must be set in child class"

        simulation_config = simulation_database(self.scenario_name)
        self.simulation_param = check_and_fix_config(simulation_config)
        self.n_timesteps_default = 10
        self.n_runs_default = 8
        self.tmpdirname = tempfile.TemporaryDirectory()  # pylint: disable=R1732

    def tearDown(self):
        self.tmpdirname.cleanup()

    def _evaluate_for_simulation_config(self, filters_configs_input):
        return evaluate_for_simulation_config(
            self.scenario_name,
            filters_configs_input,
            n_runs=self.n_runs_default,
            n_timesteps=self.n_timesteps_default,
            initial_seed=1,
            auto_warning_on_off=False,
            save_folder=self.tmpdirname.name,
        )


class TestEvalationBasics(TestEvalationBase):
    scenario_name: Optional[str] = "R2randomWalk"

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_plot_results(self):
        from pyrecest.evaluation.plot_results import plot_results

        matplotlib.pyplot.close("all")
        matplotlib.use("SVG")

        self.test_evaluate_for_simulation_config_R2_random_walk()
        filename = self._get_single_evaluation_file()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            figs, _ = plot_results(
                filename=filename,
                plot_log=False,
                plot_stds=False,
            )

        try:
            for fig in figs:
                with io.BytesIO() as buffer:
                    fig.savefig(buffer, format="png")
                    self.assertGreater(buffer.tell(), 0)
        finally:
            for fig in figs:
                fig.clf()
                matplotlib.pyplot.close(fig)

    def test_apply_log_scale_to_axes_rejects_invalid_log_shape(self):
        from pyrecest.evaluation.plot_results import apply_log_scale_to_axes

        with self.assertRaisesRegex(ValueError, "log_array"):
            apply_log_scale_to_axes([object(), object(), object()], np.zeros((3, 2)))

    def test_apply_log_scale_to_axes_rejects_wrong_axis_count(self):
        from pyrecest.evaluation.plot_results import apply_log_scale_to_axes

        with self.assertRaisesRegex(ValueError, "3 axes"):
            apply_log_scale_to_axes([object(), object()], np.zeros((2, 3)))

    def _get_single_evaluation_file(self):
        files = sorted(
            os.path.join(self.tmpdirname.name, file)
            for file in os.listdir(self.tmpdirname.name)
            if os.path.isfile(os.path.join(self.tmpdirname.name, file))
        )

        self.assertEqual(
            len(files),
            1,
            msg=(
                f"Expected exactly one evaluation file in "
                f"{self.tmpdirname.name}, got: {files}"
            ),
        )
        return files[0]

    @parameterized.expand(
        [
            (np.zeros(2),),
            (None,),
        ]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_generate_gt_R2(self, x0):
        groundtruth = generate_groundtruth(self.simulation_param, x0)

        # Test if groundtruth and its content is as expected
        self.assertEqual(np.shape(groundtruth), (self.n_timesteps_default,))
        self.assertEqual(
            np.shape(groundtruth[0]), (self.simulation_param["initial_prior"].dim,)
        )

    @parameterized.expand([(1,), (3,)])
    def test_generate_measurements(self, n_meas):
        self.simulation_param["n_meas_at_individual_time_step"] = n_meas * np.ones(
            self.n_timesteps_default, dtype=int
        )
        measurements = generate_measurements(
            np.zeros(
                (self.n_timesteps_default, self.simulation_param["initial_prior"].dim)
            ),
            self.simulation_param,
        )

        self.assertEqual(np.size(measurements), self.n_timesteps_default)
        for i in range(self.n_timesteps_default):
            self.assertEqual(
                np.atleast_2d(measurements[i]).shape,
                (
                    self.simulation_param["n_meas_at_individual_time_step"][i],
                    self.simulation_param["initial_prior"].dim,
                ),
            )

    def test_generate_measurements_uses_current_timestep_groundtruth(self):
        random.seed(0)
        simulation_param = {
            "n_timesteps": 3,
            "n_meas_at_individual_time_step": np.ones(3, dtype=int),
            "meas_noise": GaussianDistribution(
                array([0.0]),
                array([[1e-24]]),
                check_validity=False,
            ),
        }
        groundtruth = array([[1.0], [2.0], [3.0]])

        measurements = generate_measurements(groundtruth, simulation_param)

        observed = np.asarray(
            [np.asarray(measurements[t]).reshape(-1)[0] for t in range(3)],
            dtype=float,
        )
        np.testing.assert_allclose(observed, np.array([1.0, 2.0, 3.0]), atol=1e-10)

    def test_generate_measurements_preserves_singleton_measurement_axis(self):
        random.seed(0)
        simulation_param = {
            "n_timesteps": 2,
            "n_meas_at_individual_time_step": np.ones(2, dtype=int),
            "meas_noise": GaussianDistribution(
                array([0.0, 0.0]),
                array([[1e-24, 0.0], [0.0, 1e-24]]),
                check_validity=False,
            ),
        }
        groundtruth = array([[1.0, 2.0], [3.0, 4.0]])

        measurements = generate_measurements(groundtruth, simulation_param)

        self.assertEqual(tuple(measurements[0].shape), (1, 2))
        self.assertEqual(tuple(measurements[1].shape), (1, 2))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "numpy",
        reason="von Mises-Fisher sampling currently requires NumPy.",
    )
    def test_generate_measurements_does_not_mutate_directional_noise(self):
        simulation_param = {
            "n_timesteps": 2,
            "n_meas_at_individual_time_step": np.ones(2, dtype=int),
            "meas_noise": VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 5.0),
        }
        groundtruth = array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        generate_measurements(groundtruth, simulation_param)

        np.testing.assert_allclose(
            simulation_param["meas_noise"].mu,
            np.array([1.0, 0.0, 0.0]),
        )

    @parameterized.expand([("boundary",), ("within",)])
    def test_generate_measurements_eot(self, eot_sampling_style: str):
        np.random.seed(0)
        simulation_param = {
            "eot": True,
            "intensity_lambda": 0.2,
            "target_shape": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            "eot_sampling_style": eot_sampling_style,
            "n_timesteps": self.n_timesteps_default,
        }
        state_dim = 2

        measurements = generate_measurements(
            np.zeros((self.n_timesteps_default, state_dim)),
            simulation_param,
        )

        self.assertEqual(np.size(measurements), self.n_timesteps_default)
        n_meas_at_individual_time_step = array(
            [meas_at_timestep.shape[0] for meas_at_timestep in measurements]
        )
        # If one measurement at every timestep, then the number is apparently not stochastic
        self.assertFalse(all(n_meas_at_individual_time_step == 1))
        state_dim_at_individual_time_step = array(
            [meas_at_timestep.shape[-1] for meas_at_timestep in measurements]
        )
        self.assertTrue(all(state_dim_at_individual_time_step == state_dim))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ != "numpy",
        reason="hypertoroidal scenarios require NumPy-specific grid support.",
    )
    def test_determine_deviation_torus(self):
        truth = [0, 0]
        estimates = [
            array([sqrt(1 / 2), 2 * np.pi - sqrt(1 / 2)]),
            array([np.pi, np.pi]),
        ]
        deviations = determine_all_deviations(
            estimates,
            truth,
            HypertoroidalParticleFilter,
        )

        self.assertAlmostEqual(deviations[0], 1)
        self.assertAlmostEqual(deviations[1], np.sqrt(2) * np.pi)

    def test_get_axis_label(self):
        self.assertEqual(get_axis_label("time"), "Time")
        self.assertEqual(get_axis_label("xy", dim=2), ["$x_1$", "$x_2$"])

    def test_get_extract_mean(self):
        self.assertEqual(get_extract_mean(KalmanFilter), "get_point_estimate")

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_summarize_filter_results_rejects_mismatched_result_lengths(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            summarize_filter_results(
                [],
                [object()],
                get_estimate=lambda result: result,
                get_deviation=lambda estimate, truth: 0.0,
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_evaluate_for_simulation_config_R2_random_walk(self):
        filters_configs = [
            {
                "filter": KalmanFilter,
            }
        ]
        results = self._evaluate_for_simulation_config(filters_configs)
        self.assertEqual(results.shape, (self.n_runs_default, len(filters_configs)))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_evaluate_for_file(self):
        input_files = iterate_configs_and_runs(
            self.tmpdirname.name,
            [
                "run0.p",
                "run1.p",
                "run2.p",
            ],
        )
        evaluate_for_file(
            self.scenario_name,
            self.simulation_param,
            input_files,
            filters=[KalmanFilter],
            auto_warning_on_off=True,
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_configure_for_filter_rejects_unsupported_filter(self):
        with self.assertRaises(ValueError):
            configure_for_filter(dict(self.simulation_param), object)

    def test_generate_simulated_scenarios_rejects_unknown_scenario(self):
        with self.assertRaises(FileNotFoundError):
            generate_simulated_scenarios(
                n_runs=1,
                n_timesteps=1,
                scenario_names=["does-not-exist"],
                initial_seed=0,
                base_path=self.tmpdirname.name,
            )


if __name__ == "__main__":
    unittest.main()
