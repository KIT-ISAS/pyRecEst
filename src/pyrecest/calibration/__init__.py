"""Calibration helpers for asynchronous sensor-fusion workflows."""

from .bias import (
    BiasTrainingExamples,
    SensorBiasCorrectionModel,
    fit_sensor_bias_correction,
    fit_sensor_bias_correction_from_examples,
    make_bias_training_examples,
)
from .time_offset import (
    TimeOffsetFitResult,
    aggregate_time_offset_sweeps as _aggregate_time_offset_sweeps,
    apply_time_offset,
    fit_time_offset,
    interpolate_reference_values,
    make_offset_grid,
    nearest_time_indices,
    time_offset_error_summary,
    time_offset_sweep,
    _validate_error_metric,
)


def aggregate_time_offset_sweeps(sweeps, *, metric="rmse"):
    """Aggregate same-offset sweeps after normalizing the selected metric."""

    metric = _validate_error_metric(metric)
    return _aggregate_time_offset_sweeps(sweeps, metric=metric)


__all__ = [
    "BiasTrainingExamples",
    "SensorBiasCorrectionModel",
    "TimeOffsetFitResult",
    "aggregate_time_offset_sweeps",
    "apply_time_offset",
    "fit_sensor_bias_correction",
    "fit_sensor_bias_correction_from_examples",
    "fit_time_offset",
    "interpolate_reference_values",
    "make_bias_training_examples",
    "make_offset_grid",
    "nearest_time_indices",
    "time_offset_error_summary",
    "time_offset_sweep",
]
