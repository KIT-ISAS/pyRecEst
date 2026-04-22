"""Utilities for weighted ROI similarity and association-cost construction.

These helpers support session-to-session identity matching where segmented
objects are represented as dense footprint images or as sparse pixel lists such
as Suite2p ``stat.npy`` entries. In contrast to binary IoU, the weighted
metrics implemented here preserve per-pixel footprint strengths (for example
``lam``) and can therefore distinguish ROIs with identical support but
different spatial weight profiles.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


_SUPPORTED_METRICS = {"weighted_jaccard", "cosine"}


def _looks_like_sparse_coordinate_container(roi) -> bool:
    """Return ``True`` if ``roi`` looks like ``(ypix, xpix[, weights])``."""

    if not isinstance(roi, (tuple, list)) or len(roi) not in (2, 3):
        return False

    first = np.asarray(roi[0])
    second = np.asarray(roi[1])
    return first.ndim <= 1 and second.ndim <= 1


def _validate_sparse_arrays(ypix, xpix, weights) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate sparse ROI arrays and drop zero-weight pixels."""

    if ypix.shape != xpix.shape:
        raise ValueError(
            "ROI coordinate vectors 'ypix' and 'xpix' must have the same length."
        )
    if weights.shape != ypix.shape:
        raise ValueError("ROI weights must have the same length as 'ypix' and 'xpix'.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("ROI weights must be finite.")
    if np.any(weights < 0):
        raise ValueError("ROI weights must be non-negative.")

    nonzero = weights > 0
    return ypix[nonzero], xpix[nonzero], weights[nonzero]


def _extract_dense_roi(mask) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sparse coordinates from a dense 2D weighted footprint."""

    dense_mask = np.asarray(mask, dtype=float)
    if dense_mask.ndim != 2:
        raise ValueError("Dense ROI representations must be two-dimensional arrays.")
    if not np.all(np.isfinite(dense_mask)):
        raise ValueError("Dense ROI masks must contain only finite values.")
    if np.any(dense_mask < 0):
        raise ValueError("Dense ROI masks must not contain negative weights.")

    ypix, xpix = np.nonzero(dense_mask)
    weights = dense_mask[ypix, xpix].astype(float, copy=False)
    return (
        ypix.astype(np.int64, copy=False),
        xpix.astype(np.int64, copy=False),
        weights,
    )


def _extract_mapping_roi(roi, exclude_overlap=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sparse coordinates from a mapping-based ROI description."""

    if "ypix" not in roi or "xpix" not in roi:
        raise KeyError("Sparse ROI mappings must provide both 'ypix' and 'xpix' keys.")

    ypix = np.asarray(roi["ypix"], dtype=np.int64).ravel()
    xpix = np.asarray(roi["xpix"], dtype=np.int64).ravel()

    if "lam" in roi:
        weights = np.asarray(roi["lam"], dtype=float).ravel()
    elif "weights" in roi:
        weights = np.asarray(roi["weights"], dtype=float).ravel()
    else:
        weights = np.ones(ypix.shape[0], dtype=float)

    if exclude_overlap and "overlap" in roi:
        overlap = np.asarray(roi["overlap"], dtype=bool).ravel()
        if overlap.shape != ypix.shape:
            raise ValueError(
                "Suite2p-style 'overlap' flags must match the ROI pixel count."
            )
        keep = ~overlap
        ypix = ypix[keep]
        xpix = xpix[keep]
        weights = weights[keep]

    return ypix, xpix, weights


def _extract_sparse_container_roi(roi) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sparse coordinates from tuple/list ROI containers."""

    ypix = np.asarray(roi[0], dtype=np.int64).ravel()
    xpix = np.asarray(roi[1], dtype=np.int64).ravel()
    if len(roi) == 3:
        weights = np.asarray(roi[2], dtype=float).ravel()
    else:
        weights = np.ones(ypix.shape[0], dtype=float)
    return ypix, xpix, weights


def _extract_roi_pixels_and_weights(roi, exclude_overlap=False):
    """Extract sparse ROI coordinates and non-negative weights.

    Supported ROI formats are:

    * dense 2D ``numpy`` arrays, where non-zero entries define the footprint
      weights;
    * ``(ypix, xpix)`` tuples/lists;
    * ``(ypix, xpix, weights)`` tuples/lists;
    * mappings with ``"ypix"`` and ``"xpix"`` keys, optionally with ``"lam"``
      or ``"weights"`` keys, for example Suite2p ``stat.npy`` entries.
    """

    if isinstance(roi, Mapping):
        ypix, xpix, weights = _extract_mapping_roi(
            roi,
            exclude_overlap=exclude_overlap,
        )
    elif _looks_like_sparse_coordinate_container(roi):
        ypix, xpix, weights = _extract_sparse_container_roi(roi)
    else:
        ypix, xpix, weights = _extract_dense_roi(roi)

    return _validate_sparse_arrays(ypix, xpix, weights)


def _sparsify_roi(roi, exclude_overlap=False):
    """Return a sparse pixel-to-weight dictionary for ``roi``."""

    ypix, xpix, weights = _extract_roi_pixels_and_weights(
        roi,
        exclude_overlap=exclude_overlap,
    )

    sparse = {}
    for y_value, x_value, weight in zip(
        ypix.tolist(),
        xpix.tolist(),
        weights.tolist(),
    ):
        key = (int(y_value), int(x_value))
        sparse[key] = sparse.get(key, 0.0) + float(weight)
    return sparse


def _weighted_jaccard_from_sparse(sparse_a, sparse_b) -> float:
    if not sparse_a and not sparse_b:
        return 0.0

    keys = set(sparse_a) | set(sparse_b)
    numerator = 0.0
    denominator = 0.0
    for key in keys:
        weight_a = sparse_a.get(key, 0.0)
        weight_b = sparse_b.get(key, 0.0)
        numerator += min(weight_a, weight_b)
        denominator += max(weight_a, weight_b)
    return numerator / denominator if denominator > 0.0 else 0.0


def _cosine_from_sparse(sparse_a, sparse_b) -> float:
    if not sparse_a or not sparse_b:
        return 0.0

    if len(sparse_a) > len(sparse_b):
        sparse_a, sparse_b = sparse_b, sparse_a

    dot_product = sum(
        weight * sparse_b.get(key, 0.0) for key, weight in sparse_a.items()
    )
    norm_a = math.sqrt(sum(weight * weight for weight in sparse_a.values()))
    norm_b = math.sqrt(sum(weight * weight for weight in sparse_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def _similarity_from_sparse(sparse_a, sparse_b, metric):
    if metric == "weighted_jaccard":
        return _weighted_jaccard_from_sparse(sparse_a, sparse_b)
    if metric == "cosine":
        return _cosine_from_sparse(sparse_a, sparse_b)
    raise ValueError(
        f"Unsupported ROI similarity metric '{metric}'. Supported metrics are "
        f"{sorted(_SUPPORTED_METRICS)}."
    )


def _validate_similarity_metric(metric: str, parameter_name: str) -> None:
    """Validate a named ROI similarity metric."""
    del parameter_name
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported ROI similarity metric '{metric}'. Supported metrics are "
            f"{sorted(_SUPPORTED_METRICS)}."
        )


def weighted_roi_jaccard(roi_a, roi_b, exclude_overlap=False) -> float:
    """Compute a weighted Jaccard similarity between two ROIs.

    The similarity is defined as ``sum(min(a_p, b_p)) / sum(max(a_p, b_p))`` over
    the union of all active pixels. This reduces to binary IoU when all weights
    are equal to one.
    """

    sparse_a = _sparsify_roi(roi_a, exclude_overlap=exclude_overlap)
    sparse_b = _sparsify_roi(roi_b, exclude_overlap=exclude_overlap)
    return _weighted_jaccard_from_sparse(sparse_a, sparse_b)


def weighted_roi_cosine_similarity(roi_a, roi_b, exclude_overlap=False) -> float:
    """Compute cosine similarity between two weighted ROI footprints."""

    sparse_a = _sparsify_roi(roi_a, exclude_overlap=exclude_overlap)
    sparse_b = _sparsify_roi(roi_b, exclude_overlap=exclude_overlap)
    return _cosine_from_sparse(sparse_a, sparse_b)


def pairwise_roi_similarity(
    reference_rois,
    query_rois,
    metric="weighted_jaccard",
    exclude_overlap=False,
):
    """Build a pairwise ROI similarity matrix."""

    _validate_similarity_metric(metric, "metric")

    n_reference = len(reference_rois)
    n_query = len(query_rois)
    similarities = np.zeros((n_reference, n_query), dtype=float)

    sparse_reference = [
        _sparsify_roi(roi, exclude_overlap=exclude_overlap) for roi in reference_rois
    ]
    sparse_query = [
        _sparsify_roi(roi, exclude_overlap=exclude_overlap) for roi in query_rois
    ]

    for row_index, sparse_ref in enumerate(sparse_reference):
        for col_index, sparse_query_roi in enumerate(sparse_query):
            similarities[row_index, col_index] = _similarity_from_sparse(
                sparse_ref,
                sparse_query_roi,
                metric=metric,
            )

    return similarities


# pylint: disable=too-many-branches
def roi_centroid(
    roi,
    use_weights=True,
    prefer_med=True,
    exclude_overlap=False,
):
    """Return the centroid of an ROI as ``[y, x]``."""

    if prefer_med and isinstance(roi, Mapping) and "med" in roi:
        med = np.asarray(roi["med"], dtype=float).ravel()
        if med.shape != (2,):
            raise ValueError("ROI field 'med' must contain exactly two entries [y, x].")
        if not np.all(np.isfinite(med)):
            raise ValueError("ROI field 'med' must contain only finite values.")
        return med

    ypix, xpix, weights = _extract_roi_pixels_and_weights(
        roi,
        exclude_overlap=exclude_overlap,
    )
    if ypix.size == 0:
        return np.array([np.nan, np.nan], dtype=float)

    ypix = ypix.astype(float, copy=False)
    xpix = xpix.astype(float, copy=False)
    if use_weights:
        total_weight = weights.sum()
        if total_weight > 0.0:
            return np.array(
                [
                    np.dot(ypix, weights) / total_weight,
                    np.dot(xpix, weights) / total_weight,
                ],
                dtype=float,
            )

    return np.array([ypix.mean(), xpix.mean()], dtype=float)


def pairwise_centroid_distances(
    reference_rois,
    query_rois,
    use_weights=True,
    prefer_med=True,
    exclude_overlap=False,
):
    """Build a pairwise Euclidean centroid-distance matrix."""

    n_reference = len(reference_rois)
    n_query = len(query_rois)
    distances = np.full((n_reference, n_query), np.inf, dtype=float)
    if n_reference == 0 or n_query == 0:
        return distances

    reference_centroids = np.stack(
        [
            roi_centroid(
                roi,
                use_weights=use_weights,
                prefer_med=prefer_med,
                exclude_overlap=exclude_overlap,
            )
            for roi in reference_rois
        ],
        axis=0,
    ).astype(float, copy=False)
    query_centroids = np.stack(
        [
            roi_centroid(
                roi,
                use_weights=use_weights,
                prefer_med=prefer_med,
                exclude_overlap=exclude_overlap,
            )
            for roi in query_rois
        ],
        axis=0,
    ).astype(float, copy=False)

    for row_index, ref_centroid in enumerate(reference_centroids):
        if not np.all(np.isfinite(ref_centroid)):
            continue
        deltas = query_centroids - ref_centroid
        valid = np.all(np.isfinite(query_centroids), axis=1)
        distances[row_index, valid] = np.sqrt(
            np.sum(deltas[valid] * deltas[valid], axis=1)
        )

    return distances


def _validate_cost_matrix_parameters(
    footprint_metric: str,
    footprint_weight: float,
    centroid_weight: float,
    centroid_scale: float,
    max_centroid_distance,
) -> None:
    """Validate cost-matrix construction parameters."""

    _validate_similarity_metric(footprint_metric, "footprint_metric")
    if footprint_weight < 0.0:
        raise ValueError("footprint_weight must be non-negative.")
    if centroid_weight < 0.0:
        raise ValueError("centroid_weight must be non-negative.")
    if centroid_scale <= 0.0:
        raise ValueError("centroid_scale must be positive.")
    if max_centroid_distance is not None and max_centroid_distance < 0.0:
        raise ValueError("max_centroid_distance must be non-negative.")


def _compute_centroid_distance_matrix(
    reference_rois,
    query_rois,
    *,
    use_weighted_centroids,
    prefer_med,
    exclude_overlap,
):
    """Compute the centroid-distance matrix for ROI matching."""

    return pairwise_centroid_distances(
        reference_rois,
        query_rois,
        use_weights=use_weighted_centroids,
        prefer_med=prefer_med,
        exclude_overlap=exclude_overlap,
    )


def _apply_cost_matrix_gates(
    cost_matrix: np.ndarray,
    similarity_matrix: np.ndarray,
    centroid_distance_matrix,
    *,
    min_similarity,
    max_centroid_distance,
):
    """Apply hard similarity and distance gates to a cost matrix."""

    gated_cost_matrix = cost_matrix

    if min_similarity is not None:
        gated_cost_matrix = gated_cost_matrix.copy()
        gated_cost_matrix[similarity_matrix < min_similarity] = np.inf

    if max_centroid_distance is not None:
        gated_cost_matrix = gated_cost_matrix.copy()
        gated_cost_matrix[centroid_distance_matrix > max_centroid_distance] = np.inf

    return gated_cost_matrix


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
def build_roi_cost_matrix(
    reference_rois,
    query_rois,
    footprint_metric="weighted_jaccard",
    footprint_weight=1.0,
    centroid_weight=0.0,
    centroid_scale=1.0,
    max_centroid_distance=None,
    min_similarity=None,
    exclude_overlap=False,
    use_weighted_centroids=True,
    prefer_med=True,
    return_components=False,
):
    """Construct a pairwise association-cost matrix for ROI matching."""

    _validate_cost_matrix_parameters(
        footprint_metric,
        footprint_weight,
        centroid_weight,
        centroid_scale,
        max_centroid_distance,
    )

    similarity_matrix = pairwise_roi_similarity(
        reference_rois,
        query_rois,
        metric=footprint_metric,
        exclude_overlap=exclude_overlap,
    )
    cost_matrix = footprint_weight * (1.0 - similarity_matrix)

    need_centroids = centroid_weight > 0.0 or max_centroid_distance is not None
    centroid_distance_matrix = None
    if need_centroids:
        centroid_distance_matrix = _compute_centroid_distance_matrix(
            reference_rois,
            query_rois,
            use_weighted_centroids=use_weighted_centroids,
            prefer_med=prefer_med,
            exclude_overlap=exclude_overlap,
        )
        cost_matrix = cost_matrix + centroid_weight * (
            centroid_distance_matrix / centroid_scale
        )

    gated_cost_matrix = _apply_cost_matrix_gates(
        cost_matrix,
        similarity_matrix,
        centroid_distance_matrix,
        min_similarity=min_similarity,
        max_centroid_distance=max_centroid_distance,
    )

    if return_components:
        if centroid_distance_matrix is None:
            centroid_distance_matrix = np.full_like(
                gated_cost_matrix,
                np.nan,
                dtype=float,
            )
        return gated_cost_matrix, similarity_matrix, centroid_distance_matrix

    return gated_cost_matrix
