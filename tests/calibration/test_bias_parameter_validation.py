import unittest

import numpy as np

from pyrecest.calibration.bias import (
    BiasTrainingExamples,
    fit_sensor_bias_correction_from_examples,
    make_bias_training_examples,
)


class BiasParameterValidationTest(unittest.TestCase):
    def _minimal_examples(self):
        return BiasTrainingExamples(
            measured=np.array([[1.0], [2.0]]),
            reference=np.array([[0.0], [1.0]]),
            residual=np.array([[1.0], [1.0]]),
            features=np.zeros((2, 0)),
            time_delta_s=np.zeros(2),
        )

    def test_make_bias_training_examples_rejects_nonfinite_max_time_delta(self):
        for max_time_delta_s in (np.nan, np.inf):
            with self.subTest(max_time_delta_s=max_time_delta_s):
                with self.assertRaisesRegex(
                    ValueError,
                    "max_time_delta_s must be nonnegative",
                ):
                    make_bias_training_examples(
                        np.array([0.0]),
                        np.array([[1.0]]),
                        np.array([0.0]),
                        np.array([[0.0]]),
                        max_time_delta_s=max_time_delta_s,
                    )

    def test_fit_sensor_bias_correction_rejects_nonfinite_ridge_alpha(self):
        examples = self._minimal_examples()
        for ridge_alpha in (np.nan, np.inf):
            with self.subTest(ridge_alpha=ridge_alpha):
                with self.assertRaisesRegex(
                    ValueError,
                    "ridge_alpha must be nonnegative",
                ):
                    fit_sensor_bias_correction_from_examples(
                        examples,
                        ridge_alpha=ridge_alpha,
                        min_samples=1,
                    )

    def test_fit_sensor_bias_correction_rejects_nonintegral_min_samples(self):
        examples = self._minimal_examples()
        for min_samples in (0, 1.5, np.inf):
            with self.subTest(min_samples=min_samples):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_samples must be positive",
                ):
                    fit_sensor_bias_correction_from_examples(
                        examples,
                        min_samples=min_samples,
                    )


if __name__ == "__main__":
    unittest.main()
