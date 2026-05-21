"""Lightweight public type aliases used in docs, protocols, and signatures.

The aliases intentionally stay runtime-light. They document PyRecEst's shape and
semantic conventions without forcing a particular array library or shape-typing
package on users.
"""

from __future__ import annotations

from typing import Any, TypeAlias

ArrayLike: TypeAlias = Any
BackendArray: TypeAlias = Any
ScalarLike: TypeAlias = Any
StateVector: TypeAlias = Any
MeasurementVector: TypeAlias = Any
MeasurementSet: TypeAlias = Any
CovarianceMatrix: TypeAlias = Any
PrecisionMatrix: TypeAlias = Any
WeightVector: TypeAlias = Any

__all__ = [
    "ArrayLike",
    "BackendArray",
    "CovarianceMatrix",
    "MeasurementSet",
    "MeasurementVector",
    "PrecisionMatrix",
    "ScalarLike",
    "StateVector",
    "WeightVector",
]
