"""Shared helper utilities for point-set registration modules."""

from __future__ import annotations

import math
from typing import Any

from pyrecest.backend import asarray, empty, full, int64, isfinite, mean, sqrt, where, zeros
from scipy.optimize import linear_sum_assignment


def as_point_array(points, *, expected_dim: int | None = None):
    """Validate a point array with shape ``(n_points, dim)``.

    Parameters
    ----------
    points:
        Candidate point array.
    expected_dim:
        If given, require exactly this ambient dimension. Otherwise only require
        a positive point dimension.
    """
    point_array = asarray(points)
    if point_array.ndim != 2:
        raise ValueError("points must have shape (n_points, dim).")
    if point_array.shape[0] == 0:
        raise ValueError("At least one point is required.")
    if expected_dim is None:
        if point_array.shape[1] == 0:
            raise ValueError("Point dimension must be positive.")
    elif point_array.shape[1] != expected_dim:
        raise ValueError(f"Only {expected_dim}D point sets are currently supported.")
    return point_array


def validate_pair(source_points, target_points, *, expected_dim: int | None = None):
    """Validate a matched source/target point-set pair."""
    source = as_point_array(source_points, expected_dim=expected_dim)
    target = as_point_array(target_points, expected_dim=expected_dim)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    return source, target


def validate_cost_matrix_shape(cost_matrix, n_reference: int, n_moving: int) -> None:
    """Validate the shape returned by a registration association cost."""
    if cost_matrix.shape != (n_reference, n_moving):
        raise ValueError(
            "cost_function must return an array of shape (n_reference, n_moving)."
        )


def solve_gated_assignment(cost_matrix, *, max_cost: float = float("inf")):
    """Solve one-to-one assignment with optional gating and non-finite rejection."""
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


def build_matched_costs(costs, assignment):
    """Extract matched row/column indices and costs from an assignment array."""
    matched_reference_indices = where(assignment >= 0)[0]
    matched_moving_indices = assignment[matched_reference_indices]
    matched_costs = (
        costs[matched_reference_indices, matched_moving_indices]
        if matched_reference_indices.shape[0] > 0
        else empty((0,))
    )
    return matched_reference_indices, matched_moving_indices, matched_costs


def compute_rmse(matched_costs) -> float:
    """Compute the root-mean-square error of a matched-cost vector."""
    if matched_costs.shape[0] > 0:
        return float(sqrt(mean(matched_costs * matched_costs)))
    return float("inf")
