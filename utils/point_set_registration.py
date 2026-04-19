"""Point-set registration utilities for registration-aware tracking.

This module is intentionally lightweight and numpy/scipy based so it can be
used directly in tracking pipelines that operate on static landmarks such as
neuron centroids or ROI summaries across imaging sessions.

The main entry point is :func:`joint_registration_assignment`, which performs
alternating one-to-one assignment (Hungarian algorithm with optional gating) and
transform refitting. This provides a simple registration-aware matching block
that is directly useful for longitudinal neuron tracking where global drift,
rotation, or affine deformation must be estimated before data association.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

TransformModel = Literal["translation", "rigid", "affine"]
AssociationCostFn = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


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

    matrix: NDArray[np.float64]
    offset: NDArray[np.float64]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        offset = np.asarray(self.offset, dtype=float).reshape(-1)
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
        return AffineTransform(np.eye(dim, dtype=float), np.zeros(dim, dtype=float))

    def apply(self, points: ArrayLike) -> NDArray[np.float64]:
        """Apply the transform to an ``(n_points, dim)`` array of points."""
        points_array = _as_point_array(points)
        if points_array.shape[1] != self.dim:
            raise ValueError("Point dimension does not match transform dimension.")
        return (self.matrix @ points_array.T).T + self.offset

    def homogeneous_matrix(self) -> NDArray[np.float64]:
        """Return the homogeneous representation of the affine transform."""
        transform = np.eye(self.dim + 1, dtype=float)
        transform[: self.dim, : self.dim] = self.matrix
        transform[: self.dim, -1] = self.offset
        return transform


@dataclass(frozen=True)
class RegistrationResult:
    """Result of alternating registration and assignment."""

    transform: AffineTransform
    assignment: NDArray[np.int64]
    matched_reference_indices: NDArray[np.int64]
    matched_moving_indices: NDArray[np.int64]
    transformed_reference_points: NDArray[np.float64]
    matched_costs: NDArray[np.float64]
    rmse: float
    n_iterations: int
    converged: bool


def _as_point_array(points: ArrayLike) -> NDArray[np.float64]:
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2:
        raise ValueError("points must have shape (n_points, dim).")
    if points_array.shape[0] == 0:
        raise ValueError("At least one point is required.")
    if points_array.shape[1] == 0:
        raise ValueError("Point dimension must be positive.")
    return points_array



def _validate_pair(source_points: ArrayLike, target_points: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    source = _as_point_array(source_points)
    target = _as_point_array(target_points)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    return source, target



def _normalize_weights(weights: ArrayLike | None, n_points: int) -> NDArray[np.float64]:
    if weights is None:
        return np.full(n_points, 1.0 / n_points, dtype=float)
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    if weights_array.shape[0] != n_points:
        raise ValueError("weights must have length n_points.")
    if np.any(weights_array < 0.0):
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



def estimate_transform(
    source_points: ArrayLike,
    target_points: ArrayLike,
    *,
    model: TransformModel = "affine",
    weights: ArrayLike | None = None,
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

    source, target = _validate_pair(source_points, target_points)
    n_points, dim = source.shape
    min_matches = _minimum_required_matches(model, dim)
    if n_points < min_matches:
        raise ValueError(
            f"The '{model}' model requires at least {min_matches} matched points in {dim}D."
        )

    normalized_weights = _normalize_weights(weights, n_points)
    source_centroid = np.average(source, axis=0, weights=normalized_weights)
    target_centroid = np.average(target, axis=0, weights=normalized_weights)

    if model == "translation":
        return AffineTransform(np.eye(dim, dtype=float), target_centroid - source_centroid)

    if model == "rigid":
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        covariance = (normalized_weights[:, None] * source_centered).T @ target_centered
        left_singular_vectors, _, right_singular_vectors_transposed = np.linalg.svd(covariance)
        rotation = right_singular_vectors_transposed.T @ left_singular_vectors.T
        if np.linalg.det(rotation) < 0.0 and not allow_reflection:
            right_singular_vectors_transposed[-1, :] *= -1.0
            rotation = right_singular_vectors_transposed.T @ left_singular_vectors.T
        offset = target_centroid - rotation @ source_centroid
        return AffineTransform(rotation, offset)

    if model == "affine":
        design_matrix = np.concatenate([source, np.ones((n_points, 1), dtype=float)], axis=1)
        weighted_design_matrix = design_matrix * np.sqrt(normalized_weights)[:, None]
        weighted_targets = target * np.sqrt(normalized_weights)[:, None]
        coefficients, _, _, _ = np.linalg.lstsq(weighted_design_matrix, weighted_targets, rcond=None)
        matrix = coefficients[:dim, :].T
        offset = coefficients[dim, :]
        return AffineTransform(matrix, offset)

    raise ValueError(f"Unsupported transform model: {model}")



def solve_gated_assignment(cost_matrix: ArrayLike, *, max_cost: float = np.inf) -> NDArray[np.int64]:
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
    numpy.ndarray
        Integer array of shape ``(n_rows,)`` mapping each row to a column index
        or ``-1`` if the row is left unmatched.
    """

    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("cost_matrix must be two-dimensional.")
    if costs.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if costs.shape[1] == 0:
        return -np.ones((costs.shape[0],), dtype=np.int64)

    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size == 0:
        return -np.ones((costs.shape[0],), dtype=np.int64)

    if np.isfinite(max_cost):
        dummy_cost = float(max_cost)
    else:
        dummy_cost = float(finite_costs.max() + 1.0)

    padded_size = max(costs.shape)
    padded_costs = np.full((padded_size, padded_size), dummy_cost, dtype=float)
    padded_costs[: costs.shape[0], : costs.shape[1]] = costs
    row_indices, col_indices = linear_sum_assignment(padded_costs)

    assignment = -np.ones((costs.shape[0],), dtype=np.int64)
    for row_index, col_index in zip(row_indices, col_indices):
        if row_index >= costs.shape[0] or col_index >= costs.shape[1]:
            continue
        if costs[row_index, col_index] <= max_cost:
            assignment[row_index] = int(col_index)
    return assignment



def _default_cost(
    transformed_reference_points: NDArray[np.float64],
    moving_points: NDArray[np.float64],
) -> NDArray[np.float64]:
    return cdist(transformed_reference_points, moving_points, metric="euclidean")



def joint_registration_assignment(
    reference_points: ArrayLike,
    moving_points: ArrayLike,
    *,
    model: TransformModel = "affine",
    initial_transform: AffineTransform | None = None,
    max_cost: float = np.inf,
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
        reference_location = np.median(reference, axis=0)
        moving_location = np.median(moving, axis=0)
        initial_transform = AffineTransform(np.eye(dim, dtype=float), moving_location - reference_location)
    elif initial_transform.dim != dim:
        raise ValueError("initial_transform dimension must match the point dimension.")

    transform = initial_transform
    assignment = -np.ones((reference.shape[0],), dtype=np.int64)
    converged = False
    association_cost = _default_cost if cost_function is None else cost_function

    for iteration in range(1, max_iterations + 1):
        transformed_reference = transform.apply(reference)
        current_costs = np.asarray(association_cost(transformed_reference, moving), dtype=float)
        if current_costs.shape != (reference.shape[0], moving.shape[0]):
            raise ValueError(
                "cost_function must return an array of shape (n_reference, n_moving)."
            )

        new_assignment = solve_gated_assignment(current_costs, max_cost=max_cost)
        matched_reference_indices = np.flatnonzero(new_assignment >= 0)
        if matched_reference_indices.size < min_matches:
            assignment = new_assignment
            transformed_reference = transform.apply(reference)
            matched_moving_indices = assignment[matched_reference_indices]
            matched_costs = (
                current_costs[matched_reference_indices, matched_moving_indices]
                if matched_reference_indices.size > 0
                else np.empty((0,), dtype=float)
            )
            rmse = float(np.sqrt(np.mean(matched_costs**2))) if matched_costs.size > 0 else np.inf
            return RegistrationResult(
                transform=transform,
                assignment=assignment,
                matched_reference_indices=matched_reference_indices.astype(np.int64),
                matched_moving_indices=matched_moving_indices.astype(np.int64),
                transformed_reference_points=transformed_reference,
                matched_costs=matched_costs,
                rmse=rmse,
                n_iterations=iteration,
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
            float(np.linalg.norm(updated_transform.matrix - transform.matrix)),
            float(np.linalg.norm(updated_transform.offset - transform.offset)),
        )
        same_assignment = np.array_equal(new_assignment, assignment)
        transform = updated_transform
        assignment = new_assignment

        if same_assignment and parameter_change <= tolerance:
            converged = True
            break

    transformed_reference = transform.apply(reference)
    final_costs = np.asarray(association_cost(transformed_reference, moving), dtype=float)
    matched_reference_indices = np.flatnonzero(assignment >= 0)
    matched_moving_indices = assignment[matched_reference_indices]
    matched_costs = (
        final_costs[matched_reference_indices, matched_moving_indices]
        if matched_reference_indices.size > 0
        else np.empty((0,), dtype=float)
    )
    rmse = float(np.sqrt(np.mean(matched_costs**2))) if matched_costs.size > 0 else np.inf

    return RegistrationResult(
        transform=transform,
        assignment=assignment,
        matched_reference_indices=matched_reference_indices.astype(np.int64),
        matched_moving_indices=matched_moving_indices.astype(np.int64),
        transformed_reference_points=transformed_reference,
        matched_costs=matched_costs,
        rmse=rmse,
        n_iterations=iteration,
        converged=converged,
    )
