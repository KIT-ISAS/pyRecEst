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

from pyrecest.backend import (
    amax,
    asarray,
    full,
    full_like,
    int64,
    isfinite,
    nonzero,
    zeros,
)
from scipy.optimize import linear_sum_assignment


def _extract_roi_support(roi) -> set[tuple[int, int]]:
    """Return the binary support of an ROI as a set of ``(y, x)`` pixel tuples.

    Supported ROI formats are:

    * 2D dense arrays, where all non-zero entries are treated as active pixels.
    * ``(ypix, xpix)`` tuples or lists.
    * mappings with ``"ypix"`` and ``"xpix"`` keys, e.g. Suite2p ``stat`` entries.

    Parameters
    ----------
    roi:
        ROI description in one of the supported formats.

    Returns
    -------
    set[tuple[int, int]]
        Set of active pixel coordinates.
    """

    if isinstance(roi, Mapping):
        if "ypix" not in roi or "xpix" not in roi:
            raise KeyError(
                "Sparse ROI mappings must provide both 'ypix' and 'xpix' keys."
            )
        ypix = asarray(roi["ypix"], dtype=int64).ravel()
        xpix = asarray(roi["xpix"], dtype=int64).ravel()
    elif isinstance(roi, (tuple, list)) and len(roi) == 2:
        ypix = asarray(roi[0], dtype=int64).ravel()
        xpix = asarray(roi[1], dtype=int64).ravel()
    else:
        mask = asarray(roi)
        if mask.ndim != 2:
            raise ValueError(
                "Dense ROI representations must be two-dimensional arrays."
            )
        ypix, xpix = nonzero(mask)
        ypix = asarray(ypix, dtype=int64)
        xpix = asarray(xpix, dtype=int64)

    if ypix.shape != xpix.shape:
        raise ValueError("ROI coordinate vectors 'ypix' and 'xpix' must have the same length.")

    return set(zip(ypix.tolist(), xpix.tolist()))


def roi_iou(roi_a, roi_b) -> float:
    """Compute the intersection over union (IoU) between two ROIs.

    The ROIs may be provided either as dense masks or sparse pixel coordinates.
    Empty-empty pairs return ``0.0``.
    """

    support_a = _extract_roi_support(roi_a)
    support_b = _extract_roi_support(roi_b)

    if not support_a and not support_b:
        return 0.0

    intersection = len(support_a & support_b)
    union = len(support_a) + len(support_b) - intersection
    return float(intersection / union) if union > 0 else 0.0


def pairwise_iou_masks(reference_rois: Sequence, query_rois: Sequence):
    """Compute a dense pairwise IoU matrix for two ROI collections.

    Parameters
    ----------
    reference_rois:
        Sequence of ROIs associated with rows.
    query_rois:
        Sequence of ROIs associated with columns.

    Returns
    -------
    array
        Matrix with shape ``(len(reference_rois), len(query_rois))``.
    """

    n_reference = len(reference_rois)
    n_query = len(query_rois)
    iou_matrix = zeros((n_reference, n_query), dtype=float)

    if n_reference == 0 or n_query == 0:
        return iou_matrix

    reference_supports = [_extract_roi_support(roi) for roi in reference_rois]
    query_supports = [_extract_roi_support(roi) for roi in query_rois]

    for row_index, support_a in enumerate(reference_supports):
        len_a = len(support_a)
        for col_index, support_b in enumerate(query_supports):
            if not support_a and not support_b:
                continue
            intersection = len(support_a & support_b)
            union = len_a + len(support_b) - intersection
            if union > 0:
                iou_matrix[row_index, col_index] = intersection / union

    return iou_matrix


# pylint: disable=too-many-branches,too-many-locals

def assign_by_similarity_matrix(
    similarity_matrix,
    min_similarity: float = 0.0,
    num_dummy: int | None = None,
    unmatched_value: int = -1,
):
    """Solve a one-to-one assignment problem by maximizing similarity.

    The method pads the similarity matrix with dummy rows/columns and solves the
    resulting linear assignment problem using the Hungarian algorithm. Real matches
    whose similarity falls below ``min_similarity`` are disallowed; the corresponding
    row is assigned to a dummy column instead and reported as ``unmatched_value``.

    Parameters
    ----------
    similarity_matrix:
        Two-dimensional similarity matrix whose rows represent existing tracks or
        reference ROIs and whose columns represent candidate measurements or ROIs.
    min_similarity:
        Minimum similarity required for a real match.
    num_dummy:
        Number of extra dummy rows/columns. By default, ``max(n_rows, n_cols)``.
    unmatched_value:
        Value used for rows that are not matched to any real column.

    Returns
    -------
    array
        Integer vector of length ``n_rows``. Entry ``i`` contains the matched column
        index for row ``i`` or ``unmatched_value`` if no valid match exists.
    """

    similarities = asarray(similarity_matrix, dtype=float)
    if similarities.ndim != 2:
        raise ValueError("similarity_matrix must be two-dimensional.")

    n_rows, n_cols = similarities.shape
    if n_rows == 0:
        return zeros(0, dtype=int)
    if n_cols == 0:
        return full(n_rows, unmatched_value, dtype=int)

    finite_mask = isfinite(similarities)
    if not finite_mask.any():
        return full(n_rows, unmatched_value, dtype=int)

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
    padded_cost = full((padded_size, padded_size), dummy_cost, dtype=float)
    padded_cost[:n_rows, :n_cols] = cost_matrix

    row_ind, col_ind = linear_sum_assignment(padded_cost)

    assignment = full(n_rows, unmatched_value, dtype=int)
    for row_index, col_index in zip(row_ind, col_ind):
        if row_index >= n_rows:
            continue
        if col_index < n_cols and valid_mask[row_index, col_index]:
            assignment[row_index] = int(col_index)

    return assignment


# pylint: disable=too-many-positional-arguments
def associate_rois_by_iou(
    reference_rois: Sequence,
    query_rois: Sequence,
    min_iou: float = 0.0,
    num_dummy: int | None = None,
    unmatched_value: int = -1,
    return_iou_matrix: bool = False,
):
    """Associate ROIs by maximizing global IoU under one-to-one constraints.

    This is a convenience wrapper that mirrors Track2p-style session matching: build
    an ROI-overlap matrix, solve a global linear assignment problem, and reject low-
    overlap matches using ``min_iou``.

    Parameters
    ----------
    reference_rois:
        Sequence of row ROIs.
    query_rois:
        Sequence of column ROIs.
    min_iou:
        Minimum IoU required for accepting a match.
    num_dummy:
        Number of dummy rows/columns for missed assignments.
    unmatched_value:
        Value used for unmatched rows.
    return_iou_matrix:
        If ``True``, return both assignment and IoU matrix.

    Returns
    -------
    array or tuple[array, array]
        Assignment vector, optionally together with the IoU matrix.
    """

    iou_matrix = pairwise_iou_masks(reference_rois, query_rois)
    assignment = assign_by_similarity_matrix(
        iou_matrix,
        min_similarity=min_iou,
        num_dummy=num_dummy,
        unmatched_value=unmatched_value,
    )

    if return_iou_matrix:
        return assignment, iou_matrix
    return assignment
