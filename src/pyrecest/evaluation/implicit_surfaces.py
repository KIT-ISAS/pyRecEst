"""Generic helpers for evaluating implicit scalar fields and surfaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.special import ndtr

_INVALID_NUMERIC_KINDS = {"b", "S", "U", "c", "M", "m"}
_INVALID_NUMERIC_SCALAR_TYPES = (
    bool,
    np.bool_,
    str,
    bytes,
    bytearray,
    np.str_,
    np.bytes_,
    complex,
    np.complexfloating,
)


def surface_residuals(surface: Any, points: Any) -> Any:
    """Return scalar-field residuals for ``points`` via ``surface.value``."""
    if not hasattr(surface, "value"):
        raise TypeError("surface must implement value(points).")
    return surface.value(points)


def surface_gradients(surface: Any, points: Any) -> Any:
    """Return scalar-field gradients for ``points`` via ``surface.gradient``."""
    if not hasattr(surface, "gradient"):
        raise TypeError("surface must implement gradient(points).")
    return surface.gradient(points)


def surface_variances(surface: Any, points: Any) -> Any:
    """Return predictive field variances for ``points`` via ``surface.variance_at``."""
    if not hasattr(surface, "variance_at"):
        raise TypeError("surface must implement variance_at(points).")
    return surface.variance_at(points)


def surface_band_mask(values: Any, threshold: float) -> np.ndarray:
    """Return a mask for values within ``[-threshold, threshold]``."""
    threshold = _positive_float("threshold", threshold)
    array = _as_numpy_numeric_array(values, "values")
    return np.isfinite(array) & (np.abs(array) <= threshold)


def classify_inside_outside(values: Any, *, negative_inside: bool = True) -> np.ndarray:
    """Classify signed scalar-field values as inside/outside.

    Returns ``-1`` for inside, ``+1`` for outside, and ``0`` for exact zeros or
    non-finite values. Set ``negative_inside=False`` for the opposite sign
    convention.
    """
    negative_inside = _boolean_flag("negative_inside", negative_inside)
    array = _as_numpy_numeric_array(values, "values")
    labels = np.zeros(array.shape, dtype=np.int8)
    finite = np.isfinite(array)
    if negative_inside:
        labels[finite & (array < 0.0)] = -1
        labels[finite & (array > 0.0)] = 1
    else:
        labels[finite & (array > 0.0)] = -1
        labels[finite & (array < 0.0)] = 1
    return labels


def surface_band_probability_from_signed_distance(
    distance: Any,
    distance_std: Any,
    epsilon: float,
    *,
    min_std: float = 1e-4,
    normal_cdf: Callable[[Any], Any] | None = None,
) -> Any:
    """Probability that a normal signed distance lies within a surface band."""
    epsilon = _positive_float("epsilon", epsilon)
    min_std = _positive_float("min_std", min_std)
    cdf = ndtr if normal_cdf is None else normal_cdf
    distance = _as_numeric_field(distance, "distance")
    distance_std = _nonnegative_finite_numeric_field(distance_std, "distance_std")
    std = _maximum(distance_std, min_std)
    upper = (epsilon - distance) / std
    lower = (-epsilon - distance) / std
    return _clip_01(cdf(upper) - cdf(lower))


def _boolean_flag(name: str, value: bool) -> bool:
    message = f"{name} must be a boolean."
    value_array = np.asarray(value)
    if value_array.shape != ():
        raise TypeError(message)

    scalar = value_array.item()
    if not isinstance(scalar, (bool, np.bool_)):
        raise TypeError(message)
    return bool(scalar)


def _positive_float(name: str, value: float) -> float:
    message = f"{name} must be finite and positive."
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype.kind in _INVALID_NUMERIC_KINDS:
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, _INVALID_NUMERIC_SCALAR_TYPES):
        raise ValueError(message)
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(message)
    return parsed


def _as_numeric_field(values: Any, name: str) -> Any:
    if hasattr(values, "clamp"):
        return values
    return _as_numpy_numeric_array(values, name)


def _nonnegative_finite_numeric_field(values: Any, name: str) -> Any:
    if hasattr(values, "clamp"):
        if hasattr(values, "isfinite"):
            finite = values.isfinite()
            if bool((~finite).any()):
                raise ValueError(f"{name} must contain only finite values.")
        if bool((values < 0.0).any()):
            raise ValueError(f"{name} must be non-negative.")
        return values

    array = _as_numpy_numeric_array(values, name)
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def _as_numpy_numeric_array(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if array.dtype.kind in _INVALID_NUMERIC_KINDS or (
        array.dtype.kind == "O"
        and any(isinstance(item, _INVALID_NUMERIC_SCALAR_TYPES) for item in array.flat)
    ):
        raise ValueError(f"{name} must contain numeric values.")
    try:
        return array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc


def _maximum(values: Any, minimum: float) -> Any:
    if hasattr(values, "clamp"):
        return values.clamp(min=float(minimum))
    return np.maximum(values, float(minimum))


def _clip_01(values: Any) -> Any:
    if hasattr(values, "clamp"):
        return values.clamp(min=0.0, max=1.0)
    return np.clip(values, 0.0, 1.0)
