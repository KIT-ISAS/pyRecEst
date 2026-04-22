"""Utilities for overlap-aware ROI association.

These helpers are designed for session-to-session identity matching problems where
binary regions of interest (ROIs) are the primary observation. They are especially
useful for calcium-imaging pipelines that represent segmented cells either as dense
boolean masks or as sparse pixel lists such as Suite2p ``stat.npy`` entries with
``ypix``/``xpix`` coordinates.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pyrecest.backend
from pyrecest.backend import (
    amax,
    asarray,
    float64,
    full,
    full_like,
    int64,
    isfinite,
    nonzero,
    zeros,
)
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks


@dataclass(frozen=True, slots=True)
class SimilarityAssignmentResult:
    """Result of solving a similarity-based one-to-one assignment problem."""

    assignment: np.ndarray
    matched_row_indices: np.ndarray
    matched_col_indices: np.ndarray
    matched_similarities: np.ndarray
    unmatched_row_indices: np.ndarray
    unmatched_col_indices: np.ndarray

    def as_row_to_col_map(self) -> dict[int, int]:
        """Return accepted row-to-column matches as a mapping."""

        return {
            int(row_index): int(col_index)
            for row_index, col_index in zip(
                self.matched_row_indices.tolist(),
                self.matched_col_indices.tolist(),
            )
        }


@dataclass(frozen=True, slots=True)
class ROIAssociationResult:
    """Container holding the outcome of ROI association."""

    assignment: np.ndarray
    similarity_matrix: np.ndarray
    matched_reference_indices: np.ndarray
    matched_query_indices: np.ndarray
    matched_similarities: np.ndarray
    unmatched_reference_indices: np.ndarray
    unmatched_query_indices: np.ndarray
    acceptance_threshold: float | None
    threshold_method: str | None
    centroid_distance_matrix: np.ndarray | None = None

    def as_reference_to_query_map(self) -> dict[int, int]:
        """Return accepted reference-to-query matches as a mapping."""

        return {
            int(reference_index): int(query_index)
            for reference_index, query_index in zip(
                self.matched_reference_indices.tolist(),
                self.matched_query_indices.tolist(),
            )
        }


@dataclass(frozen=True, slots=True)
class _PreparedROI:
    pixels: set[tuple[int, int]]
    centroid: np.ndarray
    area: int


def _backend_not_supported(function_name: str) -> None:
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        raise NotImplementedError(
            f"{function_name} is not supported on the jax backend."
        )


def _to_numpy_array(array_like, *, dtype=None) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like.astype(dtype, copy=False) if dtype is not None else array_like
    if hasattr(array_like, "tolist"):
        return np.asarray(array_like.tolist(), dtype=dtype)
    return np.asarray(array_like, dtype=dtype)


def _extract_roi_support(roi) -> set[tuple[int, int]]:
    """Return the binary support of an ROI as a set of ``(y, x)`` pixel tuples.

    Supported ROI formats are:

    * 2D dense arrays, where all non-zero entries are treated as active pixels.
    * ``(ypix, xpix)`` tuples.
    * mappings with ``"ypix"`` and ``"xpix"`` keys, e.g. Suite2p ``stat`` entries.

    Lists are intentionally *not* treated as sparse ``(ypix, xpix)`` inputs because
    a dense mask can also be represented as a Python list with length two.
    """

    if isinstance(roi, Mapping):
        if "ypix" not in roi or "xpix" not in roi:
            raise KeyError(
                "Sparse ROI mappings must provide both 'ypix' and 'xpix' keys."
            )
        ypix = asarray(roi["ypix"], dtype=int64).ravel()
        xpix = asarray(roi["xpix"], dtype=int64).ravel()
    elif isinstance(roi, tuple) and len(roi) == 2:
        ypix = asarray(roi[0], dtype=int64).ravel()
        xpix = asarray(roi[1], dtype=int64).ravel()
    else:
        mask = asarray(roi)
        if mask.ndim != 2:
            raise ValueError(
                "Dense ROI representations must be two-dimensional arrays."
            )
        nonzero_result = nonzero(mask)
        if isinstance(nonzero_result, tuple):
            ypix, xpix = nonzero_result
        else:
            ypix = nonzero_result[:, 0]
            xpix = nonzero_result[:, 1]
        ypix = asarray(ypix, dtype=int64)
        xpix = asarray(xpix, dtype=int64)

    if ypix.shape != xpix.shape:
        raise ValueError(
            "ROI coordinate vectors 'ypix' and 'xpix' must have the same length."
        )

    return set(zip(ypix.tolist(), xpix.tolist()))


def _prepare_roi(roi) -> _PreparedROI:
    pixels = _extract_roi_support(roi)
    area = len(pixels)
    if area == 0:
        centroid = np.array([np.nan, np.nan], dtype=float)
    else:
        coords = np.asarray(tuple(pixels), dtype=float)
        centroid = coords.mean(axis=0)
    return _PreparedROI(pixels=pixels, centroid=centroid, area=area)


def _prepare_rois(rois: Sequence) -> list[_PreparedROI]:
    return [_prepare_roi(roi) for roi in rois]


def _assignment_to_result(
    assignment,
    similarity_matrix,
    *,
    unmatched_value: int,
) -> SimilarityAssignmentResult:
    similarities = _to_numpy_array(similarity_matrix, dtype=float)
    assignment_array = _to_numpy_array(assignment, dtype=int)
    matched_row_indices = np.flatnonzero(assignment_array != unmatched_value).astype(int)
    matched_col_indices = assignment_array[matched_row_indices].astype(int)
    if matched_row_indices.size == 0:
        matched_similarities = np.zeros(0, dtype=float)
    else:
        matched_similarities = similarities[
            matched_row_indices, matched_col_indices
        ].astype(float, copy=False)
    unmatched_row_indices = np.flatnonzero(assignment_array == unmatched_value).astype(int)
    unmatched_col_indices = np.setdiff1d(
        np.arange(similarities.shape[1], dtype=int),
        matched_col_indices,
        assume_unique=False,
    )
    return SimilarityAssignmentResult(
        assignment=assignment_array,
        matched_row_indices=matched_row_indices,
        matched_col_indices=matched_col_indices,
        matched_similarities=matched_similarities,
        unmatched_row_indices=unmatched_row_indices,
        unmatched_col_indices=unmatched_col_indices,
    )


def _result_to_assignment(
    result: SimilarityAssignmentResult,
    *,
    n_rows: int,
    unmatched_value: int,
) -> np.ndarray:
    assignment = np.full(n_rows, unmatched_value, dtype=int)
    if result.matched_row_indices.size > 0:
        assignment[result.matched_row_indices] = result.matched_col_indices
    return assignment


def roi_iou(roi_a, roi_b) -> float:
    """Compute the intersection over union (IoU) between two ROIs.

    The ROIs may be provided either as dense masks or sparse pixel coordinates.
    Empty-empty pairs return ``0.0``.
    """

    prepared_a = _prepare_roi(roi_a)
    prepared_b = _prepare_roi(roi_b)

    if prepared_a.area == 0 and prepared_b.area == 0:
        return 0.0

    intersection = len(prepared_a.pixels & prepared_b.pixels)
    union = prepared_a.area + prepared_b.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def roi_centroid(roi) -> np.ndarray:
    """Return the centroid of an ROI as ``[y, x]``."""

    return _prepare_roi(roi).centroid.copy()


def pairwise_centroid_distances(reference_rois: Sequence, query_rois: Sequence) -> np.ndarray:
    """Return a dense matrix of pairwise ROI centroid distances."""

    _backend_not_supported("pairwise_centroid_distances")
    prepared_reference = _prepare_rois(reference_rois)
    prepared_query = _prepare_rois(query_rois)
    n_reference = len(prepared_reference)
    n_query = len(prepared_query)
    distances = np.full((n_reference, n_query), np.inf, dtype=float)

    for row_index, reference in enumerate(prepared_reference):
        for col_index, query in enumerate(prepared_query):
            if reference.area == 0 and query.area == 0:
                distances[row_index, col_index] = 0.0
                continue
            if reference.area == 0 or query.area == 0:
                continue
            distances[row_index, col_index] = float(
                np.linalg.norm(reference.centroid - query.centroid)
            )
    return distances


def pairwise_iou_masks(
    reference_rois: Sequence,
    query_rois: Sequence,
    *,
    centroid_distance_threshold: float | None = None,
    return_centroid_distance_matrix: bool = False,
):
    """Compute a dense pairwise IoU matrix for two ROI collections.

    Parameters
    ----------
    reference_rois:
        Sequence of ROIs associated with rows.
    query_rois:
        Sequence of ROIs associated with columns.
    centroid_distance_threshold:
        Optional gating threshold in pixels. If provided, IoU values are only
        computed for ROI pairs whose centroids are within this distance.
    return_centroid_distance_matrix:
        If ``True``, also return the centroid-distance matrix.

    Returns
    -------
    array or tuple[array, numpy.ndarray]
        IoU matrix, optionally together with the centroid-distance matrix.
    """

    _backend_not_supported("pairwise_iou_masks")

    if centroid_distance_threshold is not None and centroid_distance_threshold < 0:
        raise ValueError("centroid_distance_threshold must be non-negative.")

    prepared_reference = _prepare_rois(reference_rois)
    prepared_query = _prepare_rois(query_rois)
    n_reference = len(prepared_reference)
    n_query = len(prepared_query)
    iou_matrix = zeros((n_reference, n_query), dtype=float64)
    centroid_distance_matrix = np.full((n_reference, n_query), np.inf, dtype=float)

    if n_reference == 0 or n_query == 0:
        if return_centroid_distance_matrix:
            return iou_matrix, centroid_distance_matrix
        return iou_matrix

    for row_index, reference in enumerate(prepared_reference):
        for col_index, query in enumerate(prepared_query):
            if reference.area == 0 and query.area == 0:
                centroid_distance_matrix[row_index, col_index] = 0.0
                continue
            if reference.area == 0 or query.area == 0:
                continue

            centroid_distance = float(np.linalg.norm(reference.centroid - query.centroid))
            centroid_distance_matrix[row_index, col_index] = centroid_distance
            if (
                centroid_distance_threshold is not None
                and centroid_distance > centroid_distance_threshold
            ):
                continue

            intersection = len(reference.pixels & query.pixels)
            union = reference.area + query.area - intersection
            if union > 0:
                iou_matrix[row_index, col_index] = intersection / union

    if return_centroid_distance_matrix:
        return iou_matrix, centroid_distance_matrix
    return iou_matrix


def otsu_similarity_threshold(similarities, *, nbins: int = 256) -> float:
    """Estimate a threshold using Otsu's method on one-dimensional similarities."""

    values = np.asarray(similarities, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    value_min = float(values.min())
    value_max = float(values.max())
    if value_min == value_max:
        return value_min

    histogram, bin_edges = np.histogram(
        values,
        bins=nbins,
        range=(value_min, value_max),
    )
    histogram = histogram.astype(float)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    weight_background = np.cumsum(histogram)
    weight_foreground = np.cumsum(histogram[::-1])[::-1]
    weighted_sum_background = np.cumsum(histogram * bin_centers)
    weighted_sum_foreground = np.cumsum((histogram * bin_centers)[::-1])[::-1]

    valid_mask = (weight_background > 0) & (weight_foreground > 0)
    if not np.any(valid_mask):
        return 0.0

    mean_background = np.zeros_like(bin_centers)
    mean_foreground = np.zeros_like(bin_centers)
    mean_background[valid_mask] = (
        weighted_sum_background[valid_mask] / weight_background[valid_mask]
    )
    mean_foreground[valid_mask] = (
        weighted_sum_foreground[valid_mask] / weight_foreground[valid_mask]
    )

    between_class_variance = np.zeros_like(bin_centers)
    between_class_variance[valid_mask] = (
        weight_background[valid_mask]
        * weight_foreground[valid_mask]
        * (mean_background[valid_mask] - mean_foreground[valid_mask]) ** 2
    )

    best_index = int(np.argmax(between_class_variance))
    return float(bin_centers[best_index])


def minimum_similarity_threshold(similarities, *, nbins: int = 256) -> float:
    """Estimate a threshold by locating a valley between the two strongest modes."""

    values = np.asarray(similarities, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0

    value_min = float(values.min())
    value_max = float(values.max())
    if value_min == value_max:
        return value_min

    histogram, bin_edges = np.histogram(
        values,
        bins=nbins,
        range=(value_min, value_max),
    )
    peak_indices, _ = find_peaks(histogram)

    if peak_indices.size < 2:
        return otsu_similarity_threshold(values, nbins=nbins)

    strongest_peak_indices = peak_indices[np.argsort(histogram[peak_indices])[-2:]]
    strongest_peak_indices.sort()
    left_peak, right_peak = strongest_peak_indices.tolist()

    if right_peak - left_peak <= 1:
        return otsu_similarity_threshold(values, nbins=nbins)

    valley_offset = int(np.argmin(histogram[left_peak : right_peak + 1]))
    valley_index = left_peak + valley_offset
    return float(0.5 * (bin_edges[valley_index] + bin_edges[valley_index + 1]))


# pylint: disable=too-many-branches,too-many-locals
def assign_by_similarity_matrix(
    similarity_matrix,
    min_similarity: float = 0.0,
    num_dummy: int | None = None,
    unmatched_value: int = -1,
    *,
    return_result: bool = False,
):
    """Solve a one-to-one assignment problem by maximizing similarity.

    The method pads the similarity matrix with dummy rows/columns and solves the
    resulting linear assignment problem using the Hungarian algorithm. Real matches
    whose similarity falls below ``min_similarity`` are disallowed; the corresponding
    row is assigned to a dummy column instead and reported as ``unmatched_value``.
    """

    _backend_not_supported("assign_by_similarity_matrix")

    similarities = asarray(similarity_matrix, dtype=float64)
    if similarities.ndim != 2:
        raise ValueError("similarity_matrix must be two-dimensional.")

    n_rows, n_cols = similarities.shape
    if n_rows == 0:
        empty_assignment = zeros(0, dtype=int64)
        if return_result:
            return _assignment_to_result(
                empty_assignment,
                np.zeros((0, n_cols), dtype=float),
                unmatched_value=unmatched_value,
            )
        return empty_assignment
    if n_cols == 0:
        assignment = full((n_rows,), unmatched_value, dtype=int64)
        if return_result:
            return _assignment_to_result(
                assignment,
                np.zeros((n_rows, 0), dtype=float),
                unmatched_value=unmatched_value,
            )
        return assignment

    finite_mask = isfinite(similarities)
    if not finite_mask.any():
        assignment = full((n_rows,), unmatched_value, dtype=int64)
        if return_result:
            return _assignment_to_result(
                assignment,
                _to_numpy_array(similarities, dtype=float),
                unmatched_value=unmatched_value,
            )
        return assignment

    if num_dummy is None:
        num_dummy = max(n_rows, n_cols)
    if num_dummy < 0:
        raise ValueError("num_dummy must be non-negative.")

    max_similarity = float(amax(similarities[finite_mask]))
    threshold_cost = max_similarity - float(min_similarity)
    dummy_penalty = max(
        1e-12,
        sys.float_info.epsilon * max(1.0, abs(max_similarity), abs(min_similarity)),
    )
    dummy_cost = threshold_cost + dummy_penalty

    valid_mask = finite_mask & (similarities >= min_similarity)
    cost_matrix = full_like(similarities, dummy_cost)
    cost_matrix[valid_mask] = max_similarity - similarities[valid_mask]

    padded_size = max(n_rows, n_cols) + int(num_dummy)
    padded_cost = full((padded_size, padded_size), dummy_cost, dtype=float64)
    padded_cost[:n_rows, :n_cols] = cost_matrix

    row_ind, col_ind = linear_sum_assignment(_to_numpy_array(padded_cost, dtype=float))

    assignment = full((n_rows,), unmatched_value, dtype=int64)
    for row_index, col_index in zip(row_ind, col_ind):
        if row_index >= n_rows:
            continue
        if col_index < n_cols and valid_mask[row_index, col_index]:
            assignment[row_index] = int(col_index)

    if return_result:
        return _assignment_to_result(
            assignment,
            _to_numpy_array(similarities, dtype=float),
            unmatched_value=unmatched_value,
        )
    return assignment


def _filter_matches_by_threshold(
    assignment_result: SimilarityAssignmentResult,
    *,
    threshold: float,
    n_rows: int,
    n_cols: int,
    unmatched_value: int,
) -> SimilarityAssignmentResult:
    keep_mask = assignment_result.matched_similarities >= float(threshold)
    filtered_matched_rows = assignment_result.matched_row_indices[keep_mask]
    filtered_matched_cols = assignment_result.matched_col_indices[keep_mask]
    filtered_matched_similarities = assignment_result.matched_similarities[keep_mask]

    filtered_assignment = np.full(n_rows, unmatched_value, dtype=int)
    if filtered_matched_rows.size > 0:
        filtered_assignment[filtered_matched_rows] = filtered_matched_cols

    unmatched_rows = np.setdiff1d(
        np.arange(n_rows, dtype=int),
        filtered_matched_rows,
        assume_unique=False,
    )
    unmatched_cols = np.setdiff1d(
        np.arange(n_cols, dtype=int),
        filtered_matched_cols,
        assume_unique=False,
    )

    return SimilarityAssignmentResult(
        assignment=filtered_assignment,
        matched_row_indices=filtered_matched_rows,
        matched_col_indices=filtered_matched_cols,
        matched_similarities=filtered_matched_similarities,
        unmatched_row_indices=unmatched_rows,
        unmatched_col_indices=unmatched_cols,
    )


# pylint: disable=too-many-positional-arguments
def associate_rois_by_iou(
    reference_rois: Sequence,
    query_rois: Sequence,
    min_iou: float = 0.0,
    num_dummy: int | None = None,
    unmatched_value: int = -1,
    return_iou_matrix: bool = False,
    *,
    centroid_distance_threshold: float | None = None,
    threshold_method: str | None = None,
    require_positive_match: bool = True,
    return_result: bool = False,
):
    """Associate ROIs by maximizing global IoU under one-to-one constraints.

    This mirrors Track2p-style session matching: build an ROI-overlap matrix,
    optionally gate by centroid distance, solve a global linear assignment problem,
    and reject weak matches using either a fixed threshold or an automatic
    post-assignment threshold derived from the accepted IoUs.
    """

    iou_matrix, centroid_distance_matrix = pairwise_iou_masks(
        reference_rois,
        query_rois,
        centroid_distance_threshold=centroid_distance_threshold,
        return_centroid_distance_matrix=True,
    )

    effective_min_iou = float(min_iou)
    if require_positive_match and effective_min_iou <= 0.0:
        effective_min_iou = np.nextafter(0.0, 1.0)

    assignment_result = assign_by_similarity_matrix(
        iou_matrix,
        min_similarity=effective_min_iou,
        num_dummy=num_dummy,
        unmatched_value=unmatched_value,
        return_result=True,
    )

    acceptance_threshold = None
    normalized_threshold_method = None
    if threshold_method is not None:
        normalized_threshold_method = threshold_method.lower()
        if normalized_threshold_method == "otsu":
            acceptance_threshold = otsu_similarity_threshold(
                assignment_result.matched_similarities
            )
        elif normalized_threshold_method == "min":
            acceptance_threshold = minimum_similarity_threshold(
                assignment_result.matched_similarities
            )
        else:
            raise ValueError("threshold_method must be one of None, 'otsu', or 'min'.")

        acceptance_threshold = float(max(acceptance_threshold, effective_min_iou))
        assignment_result = _filter_matches_by_threshold(
            assignment_result,
            threshold=acceptance_threshold,
            n_rows=len(reference_rois),
            n_cols=len(query_rois),
            unmatched_value=unmatched_value,
        )

    if return_result:
        result = ROIAssociationResult(
            assignment=np.asarray(assignment_result.assignment, dtype=int),
            similarity_matrix=_to_numpy_array(iou_matrix, dtype=float),
            matched_reference_indices=assignment_result.matched_row_indices,
            matched_query_indices=assignment_result.matched_col_indices,
            matched_similarities=assignment_result.matched_similarities,
            unmatched_reference_indices=assignment_result.unmatched_row_indices,
            unmatched_query_indices=assignment_result.unmatched_col_indices,
            acceptance_threshold=acceptance_threshold,
            threshold_method=normalized_threshold_method,
            centroid_distance_matrix=centroid_distance_matrix,
        )
        if return_iou_matrix:
            return result, iou_matrix
        return result

    assignment = asarray(
        _result_to_assignment(
            assignment_result,
            n_rows=len(reference_rois),
            unmatched_value=unmatched_value,
        ),
        dtype=int64,
    )
    if return_iou_matrix:
        return assignment, iou_matrix
    return assignment
