import unittest

import numpy as np
from pyrecest.models.weak_measurement import (
    MaskedLinearMeasurementModel,
    WeakDimensionMeasurementModel,
    block_diag_measurement_covariance,
    diagonal_measurement_covariance,
)


class TestWeakMeasurementInputValidation(unittest.TestCase):
    def test_diagonal_covariance_rejects_bool_and_text_stds(self):
        invalid_stds = (
            np.array([True]),
            ["1.0"],
            np.array([1.0, "2.0"], dtype=object),
            np.array([b"1.0"], dtype=object),
        )

        for stds in invalid_stds:
            with self.subTest(stds=stds):
                with self.assertRaisesRegex(
                    ValueError, "stds must contain real numeric values"
                ):
                    diagonal_measurement_covariance(stds)

    def test_block_diag_sequence_rejects_bool_and_text_stds(self):
        invalid_kwargs = (
            {"trusted_std": [True]},
            {"trusted_std": [1.0], "weak_std": ["2.0"]},
        )

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    ValueError, "stds must contain real numeric values"
                ):
                    block_diag_measurement_covariance(**kwargs)

    def test_block_diag_mapping_rejects_bool_and_text_stds(self):
        invalid_kwargs = (
            {"trusted_std": {"x": "1.0"}},
            {"trusted_std": {"x": 1.0}, "weak_std": {"range": True}},
        )

        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(
                    ValueError, "stds must contain real numeric values"
                ):
                    block_diag_measurement_covariance(**kwargs)

    def test_models_reject_bool_and_text_measurement_covariances(self):
        with self.assertRaisesRegex(
            ValueError, "measurement_noise_cov must contain real numeric values"
        ):
            MaskedLinearMeasurementModel(
                state_dim=1,
                observed_dims=[0],
                measurement_noise_cov=np.array([["1.0"]]),
            )

        with self.assertRaisesRegex(
            ValueError, "measurement_noise_cov must contain real numeric values"
        ):
            WeakDimensionMeasurementModel(
                np.eye(1), measurement_noise_cov=np.array([[True]])
            )


if __name__ == "__main__":
    unittest.main()
