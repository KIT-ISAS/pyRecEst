"""Capability protocols for implicit scalar fields and surfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .common import ArrayLike, BackendArray, SupportsInputDim


@runtime_checkable
class SupportsScalarField(SupportsInputDim, Protocol):
    """Object that can evaluate a scalar field at ambient coordinates.

    Implementations should accept a point array with shape ``(..., input_dim)``
    and return scalar values with the matching leading shape. Signed-distance
    functions and implicit surfaces are common examples, but the protocol is
    intentionally agnostic to the field semantics.
    """

    def value(self, points: ArrayLike) -> BackendArray:
        """Evaluate the scalar field at ``points``."""
        raise NotImplementedError


@runtime_checkable
class SupportsScalarFieldGradient(SupportsScalarField, Protocol):
    """Scalar field that can evaluate spatial gradients."""

    def gradient(self, points: ArrayLike) -> BackendArray:
        """Evaluate gradients at ``points`` with trailing dimension ``input_dim``."""
        raise NotImplementedError


@runtime_checkable
class SupportsProbabilisticScalarField(SupportsScalarFieldGradient, Protocol):
    """Scalar field that can evaluate predictive uncertainty."""

    def variance_at(self, points: ArrayLike) -> BackendArray:
        """Evaluate predictive variance at ``points``."""
        raise NotImplementedError
