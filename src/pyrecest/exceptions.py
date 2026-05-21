"""Shared exception types for PyRecEst user-facing errors."""

from __future__ import annotations


class PyRecEstError(Exception):
    """Base class for PyRecEst-specific exceptions."""


class BackendSupportError(PyRecEstError):
    """Base class for backend-selection and backend-capability errors."""


class BackendNotSupportedError(BackendSupportError, NotImplementedError):
    """Raised when an API is unavailable for the active backend."""


class OptionalDependencyError(PyRecEstError, ImportError):
    """Raised when an optional extra is required for a feature."""


class ValidationError(PyRecEstError, ValueError):
    """Base class for PyRecEst input validation errors."""


class ShapeError(ValidationError):
    """Raised when an input has an invalid shape."""


class DimensionMismatchError(ShapeError):
    """Raised when two inputs have incompatible dimensions."""


class NumericalStabilityError(ValidationError):
    """Raised when an operation cannot be completed stably."""


__all__ = [
    "BackendNotSupportedError",
    "BackendSupportError",
    "DimensionMismatchError",
    "NumericalStabilityError",
    "OptionalDependencyError",
    "PyRecEstError",
    "ShapeError",
    "ValidationError",
]
