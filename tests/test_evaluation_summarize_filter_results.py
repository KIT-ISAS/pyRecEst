import unittest
import warnings

import numpy as np

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.evaluation import summarize_filter_results


class TestSummarizeFilterResultsWarnings(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_run_count_warning_uses_run_axis(self):
        n_runs = 1000
        n_timesteps = 2
        groundtruths = np.empty((n_runs, n_timesteps), dtype=object)
        for index in np.ndindex(groundtruths.shape):
            groundtruths[index] = np.zeros(2)

        last_estimates = np.zeros((1, n_runs, 2))
        runtimes = np.ones((1, n_runs))
        run_failed = np.zeros((1, n_runs), dtype=bool)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            summarize_filter_results(
                scenario_config={"manifold": "Euclidean", "mtt": False},
                filter_configs=[{"name": "kf", "parameter": None}],
                runtimes=runtimes,
                groundtruths=groundtruths,
                run_failed=run_failed,
                last_estimates=last_estimates,
            )

        warning_messages = [str(warning.message) for warning in caught]
        self.assertFalse(
            any("Using less than 1000 runs" in message for message in warning_messages)
        )


if __name__ == "__main__":
    unittest.main()
