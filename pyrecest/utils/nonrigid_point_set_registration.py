"""Thin-plate-spline point-set registration utilities.

This module adds a smooth non-rigid registration primitive that is useful when
pairwise rigid or affine alignment is not expressive enough. It is intended for
registration-aware tracking problems such as longitudinal neuron identity
tracking, where ROI centroids can undergo local distortions between sessions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    any,
    array_equal,
    asarray,
    concatenate,
    empty,
    eye,
    full,
    int64,
    isfinite,
    log,
    maximum,
    mean,
    ones,
    quantile,
    sqrt,
    sum,
    where,
    zeros,
)
from pyrecest.backend import linalg
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

NonRigidAssociationCostFn = Callable


@dataclass(frozen=True)
class ThinPlateSplineTransform:
    """Two-dimensional thin-plate-spline transform.

    Parameters
    ----------
    control_points:
        Control points with shape ``(n_control, 2)``.
    weights:
        Non-rigid TPS weights with shape ``(n_control, 2)``.
    affine_coefficients:
        Affine coefficients with shape ``(3, 2)`` acting on
        ``[1, x, y]``.
    """

    control_points: object
    weights: object
    affine_coefficients: object

    def __post_init__(self) -> None:
        control_points = asarray(self.control_points)
        weights = asarray(self.weights)
        affine_coefficients = asarray(self.affine_coefficients)

        if control_points.ndim != 2:
            raise ValueError("control_points must be two-dimensional.")
        if control_points.shape[1] != 2:
            raise ValueError("ThinPlateSplineTransform currently supports 2D points only.")
        if weights.shape != control_points.shape:
            raise ValueError("weights must have the same shape as control_points.")
        if affine_coefficients.shape != (3, 2):
            raise ValueError("affine_coefficients must have shape (3, 2).")

        object.__setattr__(self, "control_points", control_points)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "affine_coefficients", affine_coefficients)

    @staticmethod
    def identity() -> "ThinPlateSplineTransform":
        """Return the identity 2D TPS transform."""
        return ThinPlateSplineTransform(
            zeros((0, 2)),
            zeros((0, 2)),
            asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        )

    @staticmethod
    def from_translation(translation) -> "ThinPlateSplineTransform":
        """Return a TPS transform representing a pure translation."""
        offset = asarray(translation).reshape(-1)
        if offset.shape[0] != 2:
            raise ValueError("translation must be two-dimensional.")
        return ThinPlateSplineTransform(
            zeros((0, 2)),
            zeros((0, 2)),
            asarray([[offset[0], offset[1]], [1.0, 0.0], [0.0, 1.0]]),
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the transform domain."""
        return 2

    def apply(self, points):
        """Apply the transform to an ``(n_points, 2)`` array of points."""
        point_array = _as_point_array(points)

        basis = zeros((point_array.shape[0], 0))
        if self.control_points.shape[0] > 0:
            distances = asarray(cdist(point_array, self.control_points, metric="euclidean"))
            basis = _tps_kernel_from_distances(distances)

        polynomial = concatenate([ones((point_array.shape[0], 1)), point_array], axis=1)
        return polynomial @ self.affine_coefficients + basis @ self.weights


@dataclass(frozen=True)
class ThinPlateSplineRegistrationResult:
    """Result of alternating TPS registration and assignment."""

    transform: ThinPlateSplineTransform
    assignment: object
    matched_reference_indices: object
    matched_moving_indices: object
    transformed_reference_points: object
    matched_costs: object
    rmse: float
    n_iterations: int
    converged: bool


def _as_point_array(points):
    point_array = asarray(points)
    if point_array.ndim != 2:
        raise ValueError("points must have shape (n_points, dim).")
    if point_array.shape[0] == 0:
        raise ValueError("At least one point is required.")
    if point_array.shape[1] != 2:
        raise ValueError("Only 2D point sets are currently supported.")
    return point_array


def _validate_pair(source_points, target_points):
    source = _as_point_array(source_points)
    target = _as_point_array(target_points)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    return source, target


def _tps_kernel_from_distances(distances, epsilon: float = 1e-12):
    squared_distances = distances * distances
    kernel = squared_distances * log(maximum(squared_distances, epsilon))
    return where(squared_distances > 0.0, kernel, 0.0)


def estimate_thin_plate_spline(
    source_points,
    target_points,
    *,
    regularization: float = 1e-3,
) -> ThinPlateSplineTransform:
    """Estimate a thin-plate-spline transform from matched 2D point pairs.

    Parameters
    ----------
    source_points, target_points:
        Arrays of shape ``(n_points, 2)`` describing matched point pairs.
    regularization:
        Non-negative ridge penalty applied to the TPS kernel matrix.
    """
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "estimate_thin_plate_spline is not supported on the JAX backend."
        )

    if regularization < 0.0:
        raise ValueError("regularization must be non-negative.")

    source, target = _validate_pair(source_points, target_points)
    n_points = source.shape[0]

    if n_points < 3:
        raise ValueError("At least three matched 2D points are required for TPS fitting.")

    kernel = _tps_kernel_from_distances(asarray(cdist(source, source, metric="euclidean")))
    polynomial = concatenate([ones((n_points, 1)), source], axis=1)

    lhs = zeros((n_points + 3, n_points + 3))
    lhs[:n_points, :n_points] = kernel + regularization * eye(n_points)
    lhs[:n_points, n_points:] = polynomial
    lhs[n_points:, :n_points] = polynomial.T

    rhs = zeros((n_points + 3, 2))
    rhs[:n_points, :] = target

    coefficients = linalg.pinv(lhs) @ rhs
    weights = coefficients[:n_points, :]
    affine_coefficients = coefficients[n_points:, :]

    return ThinPlateSplineTransform(
        control_points=source,
        weights=weights,
        affine_coefficients=affine_coefficients,
    )


def _solve_gated_assignment(cost_matrix, *, max_cost: float = float("inf")):
    costs = asarray(cost_matrix)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional.")
    if costs.shape[0] == 0:
        return zeros((0,), dtype=int64)
    if costs.shape[1] == 0:
        return zeros((costs.shape[0],), dtype=int64) - 1

    finite_mask = isfinite(costs)
    finite_costs = costs[finite_mask]
    if finite_costs.size == 0:
        return zeros((costs.shape[0],), dtype=int64) - 1

    dummy_cost = float(max_cost) if math.isfinite(max_cost) else float(finite_costs.max() + 1.0)
    padded_size = max(costs.shape[0], costs.shape[1])
    sanitized_costs = where(finite_mask, costs, dummy_cost)
    padded_costs = full((padded_size, padded_size), dummy_cost)
    padded_costs[: costs.shape[0], : costs.shape[1]] = sanitized_costs

    row_indices, col_indices = linear_sum_assignment(padded_costs)
    assignment = zeros((costs.shape[0],), dtype=int64) - 1

    for row_index, col_index in zip(row_indices, col_indices):
        if row_index >= costs.shape[0] or col_index >= costs.shape[1]:
            continue
        if isfinite(costs[row_index, col_index]) and costs[row_index, col_index] <= max_cost:
            assignment[row_index] = int(col_index)

    return assignment


def _default_cost(transformed_reference_points, moving_points):
    return cdist(transformed_reference_points, moving_points, metric="euclidean")


def _compute_rmse(matched_costs) -> float:
    if matched_costs.size > 0:
        return float(sqrt(mean(matched_costs * matched_costs)))
    return float("inf")


def joint_tps_registration_assignment(
    reference_points,
    moving_points,
    *,
    initial_transform: ThinPlateSplineTransform | None = None,
    max_cost: float = float("inf"),
    cost_function: NonRigidAssociationCostFn | None = None,
    max_iterations: int = 25,
    tolerance: float = 1e-6,
    min_matches: int = 3,
    regularization: float = 1e-3,
) -> ThinPlateSplineRegistrationResult:
    """Alternating thin-plate-spline registration and one-to-one assignment.

    This function alternates between:
      1. assigning transformed reference points to moving points using the
         Hungarian algorithm with optional gating; and
      2. refitting a smooth thin-plate-spline warp from the current matches.

    Parameters
    ----------
    reference_points:
        Landmark locations from the reference session, shape ``(n_ref, 2)``.
    moving_points:
        Landmark locations from the moving/current session, shape ``(n_moving, 2)``.
    initial_transform:
        Optional starting transform. If omitted, the transform is initialized
        with a robust median-based translation.
    max_cost:
        Optional gating threshold on the association cost.
    cost_function:
        Optional callable receiving transformed reference points and moving
        points and returning a cost matrix of shape ``(n_ref, n_moving)``.
        This allows centroid costs, ROI overlap costs, or morphology-aware
        hybrid costs to be plugged into the registration loop.
    max_iterations:
        Maximum number of alternating assignment/refit iterations.
    tolerance:
        Convergence threshold on the change of the transformed reference point set.
    min_matches:
        Minimum number of matched pairs required before refitting the TPS warp.
    regularization:
        Non-negative ridge penalty for TPS fitting.
    """
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            "joint_tps_registration_assignment is not supported on the JAX backend."
        )

    reference = _as_point_array(reference_points)
    moving = _as_point_array(moving_points)

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if min_matches < 3:
        raise ValueError("min_matches must be at least 3 for TPS fitting.")

    if initial_transform is None:
        translation = quantile(moving, 0.5, axis=0) - quantile(reference, 0.5, axis=0)
        transform = ThinPlateSplineTransform.from_translation(translation)
    else:
        if initial_transform.dim != 2:
            raise ValueError("initial_transform dimension must match the point dimension.")
        transform = initial_transform

    assignment = zeros((reference.shape[0],), dtype=int64) - 1
    converged = False
    association_cost = _default_cost if cost_function is None else cost_function

    for iteration in range(1, max_iterations + 1):
        transformed_reference = transform.apply(reference)
        current_costs = asarray(association_cost(transformed_reference, moving))
        if current_costs.shape != (reference.shape[0], moving.shape[0]):
            raise ValueError(
                "cost_function must return an array of shape (n_reference, n_moving)."
            )

        new_assignment = _solve_gated_assignment(current_costs, max_cost=max_cost)
        matched_reference_indices = where(new_assignment >= 0)[0]

        if matched_reference_indices.size < min_matches:
            assignment = new_assignment
            matched_moving_indices = assignment[matched_reference_indices]
            matched_costs = (
                current_costs[matched_reference_indices, matched_moving_indices]
                if matched_reference_indices.size > 0
                else empty((0,))
            )
            rmse = _compute_rmse(matched_costs)
            return ThinPlateSplineRegistrationResult(
                transform=transform,
                assignment=assignment,
                matched_reference_indices=matched_reference_indices.astype(int64),
                matched_moving_indices=matched_moving_indices.astype(int64),
                transformed_reference_points=transformed_reference,
                matched_costs=matched_costs,
                rmse=rmse,
                n_iterations=iteration,
                converged=False,
            )

        matched_moving_indices = new_assignment[matched_reference_indices]
        updated_transform = estimate_thin_plate_spline(
            reference[matched_reference_indices],
            moving[matched_moving_indices],
            regularization=regularization,
        )

        updated_transformed_reference = updated_transform.apply(reference)
        displacement_change = float(
            linalg.norm(updated_transformed_reference - transformed_reference)
        )
        same_assignment = bool(array_equal(new_assignment, assignment))

        transform = updated_transform
        assignment = new_assignment

        if same_assignment and displacement_change <= tolerance:
            converged = True
            break

    transformed_reference = transform.apply(reference)
    final_costs = asarray(association_cost(transformed_reference, moving))
    matched_reference_indices = where(assignment >= 0)[0]
    matched_moving_indices = assignment[matched_reference_indices]
    matched_costs = (
        final_costs[matched_reference_indices, matched_moving_indices]
        if matched_reference_indices.size > 0
        else empty((0,))
    )
    rmse = _compute_rmse(matched_costs)

    return ThinPlateSplineRegistrationResult(
        transform=transform,
        assignment=assignment,
        matched_reference_indices=matched_reference_indices.astype(int64),
        matched_moving_indices=matched_moving_indices.astype(int64),
        transformed_reference_points=transformed_reference,
        matched_costs=matched_costs,
        rmse=rmse,
        n_iterations=iteration,
        converged=converged,
    )
