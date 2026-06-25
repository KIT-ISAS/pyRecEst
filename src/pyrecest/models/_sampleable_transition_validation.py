"""Validation wiring for reusable transition-model capability flags."""

from __future__ import annotations

from typing import Any

from .likelihood import SampleableTransitionModel
from .validation import _validate_bool_flag


def _get_function_is_vectorized(model: SampleableTransitionModel) -> bool:
    return model._function_is_vectorized


def _set_function_is_vectorized(
    model: SampleableTransitionModel,
    value: Any,
) -> None:
    model._function_is_vectorized = _validate_bool_flag(
        value,
        "function_is_vectorized",
    )


def install_sampleable_transition_validation() -> None:
    """Validate ``function_is_vectorized`` after construction as well."""

    SampleableTransitionModel.function_is_vectorized = property(
        _get_function_is_vectorized,
        _set_function_is_vectorized,
        doc="Whether ``sample_next`` accepts a batch of states.",
    )
