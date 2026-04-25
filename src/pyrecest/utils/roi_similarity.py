"""Utilities for weighted ROI similarity and association-cost construction.

These helpers are aimed at session-to-session identity matching where segmented
cells are represented as dense footprint images or as sparse pixel lists such as
Suite2p ``stat.npy`` entries. In contrast to binary IoU, the weighted metrics
implemented here preserve per-pixel footprint strengths (e.g. ``lam``) and can
therefore distinguish ROIs with identical support but different spatial weight
profiles.
"""

# pylint: disable=duplicate-code
from __future__ import annotations

import math
from collections.abc import Mapping

from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    all as backend_all,
    any as backend_any,
    array,
    asarray,
    copy as backend_copy,
    dot,
    full,
    full_like,
    isfinite,
    nonzero,
    ones,
    size as backend_size,
    sqrt,
    stack,
    sum as backend_sum,
    zeros,
)


_SUPPORTED_METRICS = {"weighted_jaccard", "cosine"}


def _raise_if_unsupported_jax_backend(function_name):
    if __backend_name__ == "jax":
        raise NotImplementedError(
            f"{function_name} is not supported with the JAX backend because "
            "it builds result matrices with item assignment."
        )


def _looks_like_sparse_coordinate_container(roi) -> bool:
    """Return ``True`` if ``roi`` looks like ``(ypix, xpix[, weights])``."""

    if not isinstance(roi, (tuple, list)) or len(roi) not in (2, 3):
        return False

    first = asarray(roi[0])
    second = asarray(roi[1])
    return first.ndim <= 1 and second.ndim <= 1


# pylint: disable=too-many-branches
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-statements
def _extract_roi_pixels_and_weights(roi, exclude_overlap=False):
    """Extract sparse ROI coordinates and non-negative weights."""

    if isinstance(roi, Mapping):
        if "ypix" not in roi or "xpix" not in roi:
            raise KeyError(
                "Sparse ROI mappings must provide both 'ypix' and 'xpix' keys."
            )

        ypix = asarray(roi["ypix"], dtype=int).ravel()
        xpix = asarray(roi["xpix"], dtype=int).ravel()

        if "lam" in roi:
            weights = asarray(roi["lam"], dtype=float).ravel()
        elif "weights" in roi:
            weights = asarray(roi["weights"], dtype=float).ravel()
        else:
            weights = ones(ypix.shape[0], dtype=float)

        if exclude_overlap and "overlap" in roi:
            overlap = asarray(roi["overlap"], dtype=bool).ravel()
            if overlap.shape != ypix.shape:
                raise ValueError(
                    "Suite2p-style 'overlap' flags must match the ROI pixel count."
                )
            keep = ~overlap
            ypix = ypix[keep]
            xpix = xpix[keep]
            weights = weights[keep]

    elif _looks_like_sparse_coordinate_container(roi):
        ypix = asarray(roi[0], dtype=int).ravel()
        xpix = asarray(roi[1], dtype=int).ravel()
        if len(roi) == 3:
            weights = asarray(roi[2], dtype=float).ravel()
        else:
            weights = ones(ypix.shape[0], dtype=float)

    else:
        mask = asarray(roi, dtype=float)
        if mask.ndim != 2:
            raise ValueError(
                "Dense ROI representations must be two-dimensional arrays."
            )
        if not bool(backend_all(isfinite(mask))):
            raise ValueError("Dense ROI masks must contain only finite values.")
        if bool(backend_any(mask < 0)):
            raise ValueError("Dense ROI masks must not contain negative weights.")
        ypix, xpix = nonzero(mask)
        ypix = asarray(ypix, dtype=int)
        xpix = asarray(xpix, dtype=int)
        weights = asarray(mask[ypix, xpix], dtype=float)

    if ypix.shape != xpix.shape:
        raise ValueError(
            "ROI coordinate vectors 'ypix' and 'xpix' must have the same length."
        )
    if weights.shape != ypix.shape:
        raise ValueError("ROI weights must have the same length as 'ypix' and 'xpix'.")
    if not bool(backend_all(isfinite(weights))):
        raise ValueError("ROI weights must be finite.")
    if bool(backend_any(weights < 0)):
        raise ValueError("ROI weights must be non-negative.")

    nonzero_mask = weights > 0
    ypix = ypix[nonzero_mask]
    xpix = xpix[nonzero_mask]
    weights = weights[nonzero_mask]
    return ypix, xpix, weights


def _sparsify_roi(roi, exclude_overlap=False):
    """Return a sparse pixel->weight dictionary for ``roi``."""

    ypix, xpix, weights = _extract_roi_pixels_and_weights(
        roi, exclude_overlap=exclude_overlap
    )

    sparse = {}
    for y, x, weight in zip(ypix.tolist(), xpix.tolist(), weights.tolist()):
        key = (int(y), int(x))
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


def weighted_roi_jaccard(roi_a, roi_b, exclude_overlap=False) -> float:
    """Compute a weighted Jaccard similarity between two ROIs."""

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

    _raise_if_unsupported_jax_backend("pairwise_roi_similarity")
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported ROI similarity metric '{metric}'. Supported metrics are "
            f"{sorted(_SUPPORTED_METRICS)}."
        )

    n_reference = len(reference_rois)
    n_query = len(query_rois)
    similarities = zeros((n_reference, n_query), dtype=float)

    sparse_reference = [
        _sparsify_roi(roi, exclude_overlap=exclude_overlap) for roi in reference_rois
    ]
    sparse_query = [
        _sparsify_roi(roi, exclude_overlap=exclude_overlap) for roi in query_rois
    ]

    for i, sparse_ref in enumerate(sparse_reference):
        for j, sparse_query_roi in enumerate(sparse_query):
            similarities[i, j] = _similarity_from_sparse(
                sparse_ref, sparse_query_roi, metric=metric
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
        med = asarray(roi["med"], dtype=float).ravel()
        if med.shape != (2,):
            raise ValueError("ROI field 'med' must contain exactly two entries [y, x].")
        if not bool(backend_all(isfinite(med))):
            raise ValueError("ROI field 'med' must contain only finite values.")
        return med

    ypix, xpix, weights = _extract_roi_pixels_and_weights(
        roi, exclude_overlap=exclude_overlap
    )
    if backend_size(ypix) == 0:
        return array([math.nan, math.nan], dtype=float)

    ypix = asarray(ypix, dtype=float)
    xpix = asarray(xpix, dtype=float)
    weights = asarray(weights, dtype=float)
    if use_weights:
        total_weight = float(backend_sum(weights))
        if total_weight > 0.0:
            return array(
                [
                    float(dot(ypix, weights)) / total_weight,
                    float(dot(xpix, weights)) / total_weight,
                ],
                dtype=float,
            )

    return array(
        [
            float(backend_sum(ypix)) / float(backend_size(ypix)),
            float(backend_sum(xpix)) / float(backend_size(xpix)),
        ],
        dtype=float,
    )


def pairwise_centroid_distances(
    reference_rois,
    query_rois,
    use_weights=True,
    prefer_med=True,
    exclude_overlap=False,
):
    """Build a pairwise Euclidean centroid-distance matrix."""

    _raise_if_unsupported_jax_backend("pairwise_centroid_distances")
    n_reference = len(reference_rois)
    n_query = len(query_rois)
    distances = full((n_reference, n_query), math.inf, dtype=float)
    if n_reference == 0 or n_query == 0:
        return distances

    reference_centroids = stack(
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
    )
    query_centroids = stack(
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
    )

    valid_query_centroids = backend_all(isfinite(query_centroids), axis=1)
    for i, ref_centroid in enumerate(reference_centroids):
        if not bool(backend_all(isfinite(ref_centroid))):
            continue
        deltas = query_centroids - ref_centroid
        distances[i, valid_query_centroids] = sqrt(
            backend_sum(
                deltas[valid_query_centroids] * deltas[valid_query_centroids],
                axis=1,
            )
        )

    return distances


# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
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

    _raise_if_unsupported_jax_backend("build_roi_cost_matrix")
    if footprint_metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported ROI similarity metric '{footprint_metric}'. Supported "
            f"metrics are {sorted(_SUPPORTED_METRICS)}."
        )
    if footprint_weight < 0.0:
        raise ValueError("footprint_weight must be non-negative.")
    if centroid_weight < 0.0:
        raise ValueError("centroid_weight must be non-negative.")
    if centroid_scale <= 0.0:
        raise ValueError("centroid_scale must be positive.")
    if max_centroid_distance is not None and max_centroid_distance < 0.0:
        raise ValueError("max_centroid_distance must be non-negative.")

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
        centroid_distance_matrix = pairwise_centroid_distances(
            reference_rois,
            query_rois,
            use_weights=use_weighted_centroids,
            prefer_med=prefer_med,
            exclude_overlap=exclude_overlap,
        )
        cost_matrix = cost_matrix + centroid_weight * (
            centroid_distance_matrix / centroid_scale
        )

    if min_similarity is not None:
        cost_matrix = backend_copy(cost_matrix)
        cost_matrix[similarity_matrix < min_similarity] = math.inf

    if max_centroid_distance is not None:
        if centroid_distance_matrix is None:
            centroid_distance_matrix = pairwise_centroid_distances(
                reference_rois,
                query_rois,
                use_weights=use_weighted_centroids,
                prefer_med=prefer_med,
                exclude_overlap=exclude_overlap,
            )
        cost_matrix = backend_copy(cost_matrix)
        cost_matrix[centroid_distance_matrix > max_centroid_distance] = math.inf

    if return_components:
        if centroid_distance_matrix is None:
            centroid_distance_matrix = full_like(cost_matrix, math.nan, dtype=float)
        return cost_matrix, similarity_matrix, centroid_distance_matrix

    return cost_matrix


__all__ = [
    "build_roi_cost_matrix",
    "pairwise_centroid_distances",
    "pairwise_roi_similarity",
    "roi_centroid",
    "weighted_roi_cosine_similarity",
    "weighted_roi_jaccard",
]
