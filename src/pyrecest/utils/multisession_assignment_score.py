"""Score-native conveniences for multi-session association."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    all as all_,
    asarray,
    copy,
    full,
    isfinite,
    where,
)

from ._multisession_assignment_common import (
    BackendArray,
    PairwiseCostsInput,
    SessionSizesInput,
    TrackInput,
    infer_and_validate_session_sizes as _infer_and_validate_session_sizes,
    infer_track_session_sizes as _infer_track_session_sizes,
    iter_track_items as _iter_track_items,
    normalize_pairwise_costs as _normalize_pairwise_costs,
    normalize_session_sizes as _normalize_session_sizes,
    session_gap as _session_gap,
)
from .multisession_assignment import (
    MultiSessionAssignmentResult,
    solve_multisession_assignment,
)


def _default_score_to_cost(scores: BackendArray) -> BackendArray:
    return -asarray(scores, dtype=float)


def tracks_to_index_matrix(
    tracks: list[TrackInput],
    session_sizes: SessionSizesInput | None = None,
    *,
    fill_value: int = -1,
):
    """Convert tracks to a dense ``track x session`` ROI-index matrix."""
    assert __backend_name__ != "jax", "Not supported on JAX backend"

    inferred_sizes, max_session_index = _infer_track_session_sizes(
        tracks,
        session_sizes,
        require_unique_sessions=True,
    )
    matrix = full((len(tracks), max_session_index + 1), fill_value, dtype=int)
    for track_index, track in enumerate(tracks):
        for session_index, detection_index in _iter_track_items(track):
            matrix[track_index, session_index] = detection_index
    return matrix


def solve_multisession_assignment_from_similarity(  # pylint: disable=too-many-arguments,too-many-locals
    pairwise_scores: PairwiseCostsInput,
    session_sizes: SessionSizesInput | None = None,
    *,
    min_score: float | None = None,
    max_gap: int | None = None,
    gap_penalty: float = 0.0,
    start_cost: float = 0.0,
    end_cost: float = 0.0,
    score_to_cost: Callable[[BackendArray], BackendArray] | None = None,
) -> MultiSessionAssignmentResult:
    """Score-native wrapper around :func:`solve_multisession_assignment`."""
    assert __backend_name__ != "jax", "Not supported on JAX backend"

    if min_score is not None and not math.isfinite(min_score):
        raise ValueError("min_score must be finite.")
    max_gap = _validate_max_gap(max_gap)

    normalized_pairwise_scores = _normalize_pairwise_costs(pairwise_scores)
    normalized_session_sizes = _normalize_session_sizes(session_sizes)
    session_sizes_map = _infer_and_validate_session_sizes(
        normalized_pairwise_scores,
        normalized_session_sizes,
    )
    if score_to_cost is None:
        score_to_cost = _default_score_to_cost

    session_positions = {
        session_idx: position
        for position, session_idx in enumerate(sorted(session_sizes_map))
    }
    transformed_pairwise_costs = _transform_pairwise_scores(
        normalized_pairwise_scores,
        session_positions,
        min_score=min_score,
        max_gap=max_gap,
        score_to_cost=score_to_cost,
    )

    return solve_multisession_assignment(
        transformed_pairwise_costs,
        session_sizes=session_sizes_map,
        start_cost=start_cost,
        end_cost=end_cost,
        gap_penalty=gap_penalty,
    )


def stitch_tracks_from_pairwise_scores(
    pairwise_scores: PairwiseCostsInput,
    session_sizes: SessionSizesInput | None = None,
    **kwargs,
) -> MultiSessionAssignmentResult:
    """Track2p-style alias for the score-native wrapper."""
    return solve_multisession_assignment_from_similarity(
        pairwise_scores,
        session_sizes=session_sizes,
        **kwargs,
    )


def _validate_max_gap(max_gap: int | None) -> int | None:
    if max_gap is None:
        return None

    original_max_gap = max_gap
    normalized_max_gap = int(max_gap)
    if normalized_max_gap != original_max_gap:
        raise ValueError("max_gap must be an integer.")
    if normalized_max_gap < 0:
        raise ValueError("max_gap must be non-negative.")
    return normalized_max_gap


def _transform_pairwise_scores(
    pairwise_scores: Mapping[tuple[int, int], BackendArray],
    session_positions: Mapping[int, int],
    *,
    min_score: float | None,
    max_gap: int | None,
    score_to_cost: Callable[[BackendArray], BackendArray],
) -> dict[tuple[int, int], BackendArray]:
    transformed_pairwise_costs: dict[tuple[int, int], BackendArray] = {}
    for (source_session, target_session), score_matrix in pairwise_scores.items():
        gap = _session_gap(source_session, target_session, session_positions)
        if max_gap is not None and gap > max_gap:
            continue

        score_matrix_array = asarray(score_matrix, dtype=float)
        finite_mask = isfinite(score_matrix_array)
        if min_score is not None:
            finite_mask = finite_mask & (score_matrix_array >= float(min_score))

        safe_scores = where(isfinite(score_matrix_array), score_matrix_array, 0.0)
        cost_matrix = asarray(score_to_cost(safe_scores), dtype=float)
        if cost_matrix.shape != score_matrix_array.shape:
            raise ValueError("score_to_cost must preserve the input matrix shape.")

        cost_matrix = copy(cost_matrix)
        cost_matrix[~finite_mask] = float("inf")
        if not all_(isfinite(cost_matrix[finite_mask])):
            raise ValueError(
                "score_to_cost must return finite costs for admissible scores."
            )
        transformed_pairwise_costs[(source_session, target_session)] = cost_matrix
    return transformed_pairwise_costs


def _result_to_index_matrix(
    self,
    session_sizes: SessionSizesInput | None = None,
    *,
    fill_value: int = -1,
):
    return tracks_to_index_matrix(
        self.tracks,
        session_sizes=session_sizes,
        fill_value=fill_value,
    )


MultiSessionAssignmentResult.to_index_matrix = _result_to_index_matrix  # type: ignore[attr-defined]
