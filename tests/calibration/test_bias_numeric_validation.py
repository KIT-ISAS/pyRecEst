import numpy as np
import pytest

from pyrecest.calibration.bias import (
    SensorBiasCorrectionModel,
    make_bias_training_examples,
)


def _unit_feature_model_kwargs():
    return {
        "target_dim": 1,
        "feature_dim": 1,
        "intercept": np.array([0.0]),
        "coefficients": np.array([[1.0]]),
        "feature_mean": np.array([0.0]),
        "feature_scale": np.array([1.0]),
        "residual_std": np.array([0.0]),
        "training_count": 2,
        "ridge_alpha": 0.0,
    }


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("intercept", np.array(["0.0"])),
        ("coefficients", np.array([[True]])),
        ("feature_mean", np.array(["0.0"])),
        ("feature_scale", np.array([False], dtype=object)),
        ("residual_std", np.array(["0.0"], dtype=object)),
    ],
)
def test_model_rejects_bool_and_text_array_fields(field_name, invalid_value):
    kwargs = _unit_feature_model_kwargs()
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError, match=rf"{field_name} must contain numeric values"):
        SensorBiasCorrectionModel(**kwargs)


def test_from_dict_rejects_text_array_fields_before_float_coercion():
    payload = SensorBiasCorrectionModel(**_unit_feature_model_kwargs()).to_dict()
    payload["intercept"] = ["0.0"]

    with pytest.raises(ValueError, match="intercept must contain numeric values"):
        SensorBiasCorrectionModel.from_dict(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("measurement_times_s", np.array(["0.0", "1.0"])),
        ("measurement_values", np.array([[True], [False]])),
        ("reference_times_s", np.array(["0.0", "1.0"])),
        ("reference_values", np.array([["1.0"], ["2.0"]], dtype=object)),
        ("feature_values", np.array([[True], [False]], dtype=object)),
    ],
)
def test_make_bias_training_examples_rejects_bool_and_text_inputs(field_name, invalid_value):
    kwargs = {
        "measurement_times_s": np.array([0.0, 1.0]),
        "measurement_values": np.array([[1.0], [2.0]]),
        "reference_times_s": np.array([0.0, 1.0]),
        "reference_values": np.array([[1.5], [2.5]]),
        "feature_values": np.array([[0.0], [1.0]]),
    }
    kwargs[field_name] = invalid_value

    with pytest.raises(ValueError, match=rf"{field_name} must contain numeric values"):
        make_bias_training_examples(**kwargs)
