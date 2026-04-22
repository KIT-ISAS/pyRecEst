"""Shared helpers for rigid and non-rigid point-set registration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    asarray,
    cast,
    empty,
    full,
    int64,
    isfinite,
    mean,
    sqrt,
    where,
    zeros,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class RegistrationResultBase:  # pylint: disable=too-many-instance-attributes
    """Shared fields for registration result containers."""

    assignment: Any
    matched_reference_indices: Any
    matched_moving_indices: Any
    transformed_reference_points: Any
    matched_costs: Any
    rmse: float
    n_iterations: int
    converged: bool


@dataclass(frozen=True)
class MatchSummary:
    """Matched indices, costs, and RMSE diagnostics for an assignment."""

    matched_reference_indices: Any
    matched_moving_indices: Any
    matched_costs: Any
    rmse: float


def as_point_array(points, *, expected_dim: int | None = None, expected_dim_error: str | None = None):
    """Validate a point array and optionally enforce its ambient dimension."""
    point_array = asarray(points)
    if point_array.ndim != 2:
        raise ValueError("points must have shape (n_points, dim).")
    if point_array.shape[0] == 0:
        raise ValueError("At least one point is required.")
    if expected_dim is None:
        if point_array.shape[1] == 0:
            raise ValueError("Point dimension must be positive.")
    elif point_array.shape[1] != expected_dim:
        raise ValueError(expected_dim_error or f"points must have dimension {expected_dim}.")
    return point_array


def validate_pair(
    source_points,
    target_points,
    *,
    expected_dim: int | None = None,
    expected_dim_error: str | None = None,
):
    """Validate a pair of matched point arrays."""
    source = as_point_array(
        source_points,
        expected_dim=expected_dim,
        expected_dim_error=expected_dim_error,
    )
    target = as_point_array(
        target_points,
        expected_dim=expected_dim,
        expected_dim_error=expected_dim_error,
    )
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    return source, target


def validate_cost_matrix(cost_matrix, *, n_reference: int, n_moving: int):
    """Validate the shape of an association cost matrix."""
    costs = asarray(cost_matrix)
    if costs.shape != (n_reference, n_moving):
        raise ValueError("cost_function must return an array of shape (n_reference, n_moving).")
    return costs


def evaluate_registration_costs(transform, reference, moving, association_cost):
    """Transform reference points and evaluate the association cost matrix."""
    transformed_reference = transform.apply(reference)
    current_costs = validate_cost_matrix(
        association_cost(transformed_reference, moving),
        n_reference=reference.shape[0],
        n_moving=moving.shape[0],
    )
    return transformed_reference, current_costs


def solve_gated_assignment(cost_matrix, *, max_cost: float = float("inf")):
    """Solve one-to-one assignment with optional gating."""
    costs = asarray(cost_matrix)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional.")
    if costs.shape[0] == 0:
        return zeros((0,), dtype=int64)
    if costs.shape[1] == 0:
        return zeros((costs.shape[0],), dtype=int64) - 1

    finite_mask = isfinite(costs)
    finite_costs = costs[finite_mask]
    if finite_costs.shape[0] == 0:
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


def default_cost(transformed_reference_points, moving_points):
    """Default Euclidean association cost."""
    return cdist(transformed_reference_points, moving_points, metric="euclidean")


def compute_rmse(matched_costs) -> float:
    """Compute the RMSE over matched costs."""
    if matched_costs.shape[0] > 0:
        return float(sqrt(mean(matched_costs * matched_costs)))
    return float("inf")


def summarize_assignment(assignment, costs) -> MatchSummary:
    """Extract matched row/column indices and diagnostics from an assignment."""
    matched_reference_indices = where(assignment >= 0)[0]
    matched_moving_indices = assignment[matched_reference_indices]
    matched_costs = (
        costs[matched_reference_indices, matched_moving_indices]
        if matched_reference_indices.shape[0] > 0
        else empty((0,))
    )
    return MatchSummary(
        matched_reference_indices=cast(matched_reference_indices, int64),
        matched_moving_indices=cast(matched_moving_indices, int64),
        matched_costs=matched_costs,
        rmse=compute_rmse(matched_costs),
    )

# pylint: disable=too-many-arguments
def build_registration_result(
    result_type,
    *,
    transform,
    assignment,
    transformed_reference_points,
    costs,
    iteration: int,
    converged: bool,
):
    """Construct a registration result object from common bookkeeping."""
    summary = summarize_assignment(assignment, costs)
    return result_type(
        transform=transform,
        assignment=assignment,
        matched_reference_indices=summary.matched_reference_indices,
        matched_moving_indices=summary.matched_moving_indices,
        transformed_reference_points=transformed_reference_points,
        matched_costs=summary.matched_costs,
        rmse=summary.rmse,
        n_iterations=iteration,
        converged=converged,
    )
