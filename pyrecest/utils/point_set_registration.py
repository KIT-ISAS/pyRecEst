"""Point-set registration utilities for registration-aware tracking.

This module is intentionally lightweight so it can be used directly in tracking
pipelines that operate on static landmarks such as neuron centroids or ROI
summaries across imaging sessions.

The main entry point is :func:`joint_registration_assignment`, which performs
alternating one-to-one assignment (Hungarian algorithm with optional gating) and
transform refitting. This provides a simple registration-aware matching block
that is directly useful for longitudinal neuron tracking where global drift,
rotation, or affine deformation must be estimated before data association.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal

import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    any,
    array_equal,
    asarray,
    cast,
    concatenate,
    empty,
    eye,
    full,
    int64,
    isfinite,
    linalg,
    mean,
    ones,
    quantile,
    sqrt,
    sum,
    where,
    zeros,
)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

TransformModel = Literal["translation", "rigid", "affine"]
AssociationCostFn = Callable


@dataclass(frozen=True)
class AffineTransform:
    """An affine transform represented by its linear part and translation.

    Parameters
    ----------
    matrix:
        Linear part of shape ``(dim, dim)``.
    offset:
        Translation vector of shape ``(dim,)``.
    """

    matrix: Any
    offset: Any

    def __post_init__(self) -> None:
        matrix = asarray(self.matrix)
        offset = asarray(self.offset).reshape(-1)
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")
        if offset.shape[0] != matrix.shape[0]:
            raise ValueError("offset dimension must match matrix dimension.")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "offset", offset)

    @property
    def dim(self) -> int:
        return self.offset.shape[0]

    @staticmethod
    def identity(dim: int) -> "AffineTransform":
        """Return the identity transform in ``dim`` dimensions."""
        if dim <= 0:
            raise ValueError("dim must be positive.")
        return AffineTransform(eye(dim), zeros(dim))

    def apply(self, points) -> Any:
        """Apply the transform to an ``(n_points, dim)`` array of points."""
        points_array = _as_point_array(points)
        if points_array.shape[1] != self.dim:
            raise ValueError("Point dimension does not match transform dimension.")
        return (self.matrix @ points_array.T).T + self.offset

    def homogeneous_matrix(self) -> Any:
        """Return the homogeneous representation of the affine transform."""
        transform = eye(self.dim + 1)
        transform[: self.dim, : self.dim] = self.matrix
        transform[: self.dim, -1] = self.offset
        return transform


@dataclass(frozen=True)
class RegistrationResult:  # pylint: disable=too-many-instance-attributes
    """Result of alternating registration and assignment."""

    transform: AffineTransform
    assignment: Any
    matched_reference_indices: Any
    matched_moving_indices: Any
    transformed_reference_points: Any
    matched_costs: Any
    rmse: float
    n_iterations: int
    converged: bool


def _as_point_array(points):
    points_array = asarray(points)
    if points_array.ndim != 2:
        raise ValueError("points must have shape (n_points, dim).")
    if points_array.shape[0] == 0:
        raise ValueError("At least one point is required.")
    if points_array.shape[1] == 0:
        raise ValueError("Point dimension must be positive.")
    return points_array


def _validate_pair(source_points, target_points):
    source = _as_point_array(source_points)
    target = _as_point_array(target_points)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    return source, target


def _normalize_weights(weights, n_points):
    if weights is None:
        return full((n_points,), 1.0 / n_points)
    weights_array = asarray(weights).reshape(-1)
    if weights_array.shape[0] != n_points:
        raise ValueError("weights must have length n_points.")
    if any(weights_array < 0.0):
        raise ValueError("weights must be non-negative.")
    weight_sum = float(weights_array.sum())
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return weights_array / weight_sum


def _minimum_required_matches(model: TransformModel, dim: int) -> int:
    if model == "translation":
        return 1
    if model == "rigid":
        return max(2, dim)
    if model == "affine":
        return dim + 1
    raise ValueError(f"Unsupported transform model: {model}")


def estimate_transform(  # pylint: disable=too-many-locals
    source_points,
    target_points,
    *,
    model: TransformModel = "affine",
    weights=None,
    allow_reflection: bool = False,
) -> AffineTransform:
    """Estimate a transform from matched source/target point pairs.

    Parameters
    ----------
    source_points, target_points:
        Arrays of shape ``(n_points, dim)`` describing matched point pairs.
    model:
        ``"translation"``, ``"rigid"``, or ``"affine"``.
    weights:
        Optional non-negative per-point weights.
    allow_reflection:
        Only relevant for the rigid model. If ``False`` the returned rotation is
        constrained to have determinant ``+1``.
    """
    assert pyrecest.backend.__backend_name__ != "jax", "Not supported for the JAX backend."

    source, target = _validate_pair(source_points, target_points)
    n_points, dim = source.shape
    min_matches = _minimum_required_matches(model, dim)
    if n_points < min_matches:
        raise ValueError(
            f"The '{model}' model requires at least {min_matches} matched points in {dim}D."
        )

    normalized_weights = _normalize_weights(weights, n_points)
    source_centroid = sum(normalized_weights[:, None] * source, axis=0)
    target_centroid = sum(normalized_weights[:, None] * target, axis=0)

    if model == "translation":
        return AffineTransform(eye(dim), target_centroid - source_centroid)

    if model == "rigid":
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        covariance = (normalized_weights[:, None] * source_centered).T @ target_centered
        left_singular_vectors, _, right_singular_vectors_transposed = linalg.svd(covariance)
        rotation = right_singular_vectors_transposed.T @ left_singular_vectors.T
        if linalg.det(rotation) < 0.0 and not allow_reflection:
            right_singular_vectors_transposed[-1, :] *= -1.0
            rotation = right_singular_vectors_transposed.T @ left_singular_vectors.T
        offset = target_centroid - rotation @ source_centroid
        return AffineTransform(rotation, offset)

    if model == "affine":
        design_matrix = concatenate([source, ones((n_points, 1))], axis=1)
        weighted_design_matrix = design_matrix * sqrt(normalized_weights)[:, None]
        weighted_targets = target * sqrt(normalized_weights)[:, None]
        coefficients = linalg.pinv(weighted_design_matrix) @ weighted_targets
        matrix = coefficients[:dim, :].T
        offset = coefficients[dim, :]
        return AffineTransform(matrix, offset)

    raise ValueError(f"Unsupported transform model: {model}")


def solve_gated_assignment(cost_matrix, *, max_cost: float = float("inf")):
    """Solve one-to-one assignment with optional gating.

    Parameters
    ----------
    cost_matrix:
        Array of shape ``(n_rows, n_cols)``.
    max_cost:
        Matches with cost strictly larger than this value are rejected and
        encoded as ``-1`` in the output.

    Returns
    -------
    Integer array of shape ``(n_rows,)`` mapping each row to a column index
    or ``-1`` if the row is left unmatched.
    """
    assert pyrecest.backend.__backend_name__ != "jax", "Not supported for the JAX backend."

    costs = asarray(cost_matrix)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional.")
    if costs.shape[0] == 0:
        return zeros((0,), dtype=int64)
    if costs.shape[1] == 0:
        return zeros((costs.shape[0],), dtype=int64) - 1

    finite_mask = isfinite(costs)
    finite_costs = costs[finite_mask]
    if len(finite_costs) == 0:
        return zeros((costs.shape[0],), dtype=int64) - 1

    if math.isfinite(max_cost):
        dummy_cost = float(max_cost)
    else:
        dummy_cost = float(finite_costs.max() + 1.0)

    padded_size = max(costs.shape)
    padded_costs = full((padded_size, padded_size), dummy_cost)
    padded_costs[: costs.shape[0], : costs.shape[1]] = costs
    row_indices, col_indices = linear_sum_assignment(padded_costs)

    assignment = zeros((costs.shape[0],), dtype=int64) - 1
    for row_index, col_index in zip(row_indices, col_indices):
        if row_index >= costs.shape[0] or col_index >= costs.shape[1]:
            continue
        if costs[row_index, col_index] <= max_cost:
            assignment[row_index] = int(col_index)
    return assignment


def _default_cost(transformed_reference_points, moving_points):
    return cdist(transformed_reference_points, moving_points, metric="euclidean")


def _compute_rmse(matched_costs) -> float:
    if len(matched_costs) > 0:
        return float(sqrt(mean(matched_costs**2)))
    return float("inf")


def _build_registration_result(
    transform: AffineTransform,
    assignment: Any,
    transformed_reference_points: Any,
    costs: Any,
    *,
    iteration: int,
    converged: bool,
) -> RegistrationResult:
    matched_reference_indices = where(assignment >= 0)[0]
    matched_moving_indices = assignment[matched_reference_indices]
    matched_costs = (
        costs[matched_reference_indices, matched_moving_indices]
        if len(matched_reference_indices) > 0
        else empty((0,))
    )
    rmse = _compute_rmse(matched_costs)
    return RegistrationResult(
        transform=transform,
        assignment=assignment,
        matched_reference_indices=cast(matched_reference_indices, int64),
        matched_moving_indices=cast(matched_moving_indices, int64),
        transformed_reference_points=transformed_reference_points,
        matched_costs=matched_costs,
        rmse=rmse,
        n_iterations=iteration,
        converged=converged,
    )


def joint_registration_assignment(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    reference_points,
    moving_points,
    *,
    model: TransformModel = "affine",
    initial_transform: AffineTransform | None = None,
    max_cost: float = float("inf"),
    cost_function: AssociationCostFn | None = None,
    max_iterations: int = 25,
    tolerance: float = 1e-8,
    min_matches: int | None = None,
    allow_reflection: bool = False,
) -> RegistrationResult:
    """Alternating registration and one-to-one assignment.

    This function alternates between:

    1. assigning transformed reference points to moving points using the
       Hungarian algorithm with optional gating; and
    2. refitting the specified transform model using the current matches.

    Parameters
    ----------
    reference_points:
        Landmark locations from the reference session, shape ``(n_ref, dim)``.
    moving_points:
        Landmark locations from the moving/current session, shape ``(n_moving, dim)``.
    model:
        Registration model to fit: ``"translation"``, ``"rigid"``, or ``"affine"``.
    initial_transform:
        Optional starting transform. If omitted, a coordinate-wise median
        alignment is used as initialization. For challenging partial-overlap or
        high-outlier cases, providing an external coarse registration is
        recommended.
    max_cost:
        Optional gating threshold on the association cost.
    cost_function:
        Optional callable that receives transformed reference points and moving
        points and returns a cost matrix of shape ``(n_ref, n_moving)``. This
        makes it easy to plug in ROI-overlap or morphology-aware costs on top of
        centroid registration.
    max_iterations:
        Maximum number of alternating assignment/refit iterations.
    tolerance:
        Convergence threshold on the change of the affine parameters.
    min_matches:
        Minimum number of matched pairs required before refitting. Defaults to
        the identifiability threshold of the chosen model.
    allow_reflection:
        Passed through to :func:`estimate_transform` for the rigid model.
    """
    assert pyrecest.backend.__backend_name__ != "jax", "Not supported for the JAX backend."

    reference = _as_point_array(reference_points)
    moving = _as_point_array(moving_points)
    if reference.shape[1] != moving.shape[1]:
        raise ValueError("reference_points and moving_points must have the same dimension.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    dim = reference.shape[1]
    if min_matches is None:
        min_matches = _minimum_required_matches(model, dim)
    if min_matches <= 0:
        raise ValueError("min_matches must be positive.")

    if initial_transform is None:
        reference_location = quantile(reference, 0.5, axis=0)
        moving_location = quantile(moving, 0.5, axis=0)
        initial_transform = AffineTransform(eye(dim), moving_location - reference_location)
    elif initial_transform.dim != dim:
        raise ValueError("initial_transform dimension must match the point dimension.")

    transform = initial_transform
    assignment = zeros((reference.shape[0],), dtype=int64) - 1
    converged = False
    iteration = 0
    association_cost = _default_cost if cost_function is None else cost_function

    for iteration in range(1, max_iterations + 1):
        transformed_reference = transform.apply(reference)
        current_costs = asarray(association_cost(transformed_reference, moving))
        if current_costs.shape != (reference.shape[0], moving.shape[0]):
            raise ValueError(
                "cost_function must return an array of shape (n_reference, n_moving)."
            )

        new_assignment = solve_gated_assignment(current_costs, max_cost=max_cost)
        matched_reference_indices = where(new_assignment >= 0)[0]
        if len(matched_reference_indices) < min_matches:
            assignment = new_assignment
            return _build_registration_result(
                transform,
                assignment,
                transformed_reference,
                current_costs,
                iteration=iteration,
                converged=False,
            )

        matched_moving_indices = new_assignment[matched_reference_indices]
        updated_transform = estimate_transform(
            reference[matched_reference_indices],
            moving[matched_moving_indices],
            model=model,
            allow_reflection=allow_reflection,
        )

        parameter_change = max(
            float(linalg.norm(updated_transform.matrix - transform.matrix)),
            float(linalg.norm(updated_transform.offset - transform.offset)),
        )
        same_assignment = bool(array_equal(new_assignment, assignment))
        transform = updated_transform
        assignment = new_assignment

        if same_assignment and parameter_change <= tolerance:
            converged = True
            break

    transformed_reference = transform.apply(reference)
    final_costs = asarray(association_cost(transformed_reference, moving))
    return _build_registration_result(
        transform,
        assignment,
        transformed_reference,
        final_costs,
        iteration=iteration,
        converged=converged,
    )
