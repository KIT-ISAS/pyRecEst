"""Score-native conveniences for multi-session association."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from pyrecest.backend import __backend_name__, full  # pylint: disable=no-name-in-module

from .multisession_assignment import (
    MultiSessionAssignmentResult,
    PairwiseCostsInput,
    SessionSizesInput,
    TrackInput,
    _infer_and_validate_session_sizes,
    _iter_track_items,
    _normalize_pairwise_costs,
    _normalize_session_sizes,
    solve_multisession_assignment,
)


def _default_score_to_cost(scores: np.ndarray) -> np.ndarray:
    return -np.asarray(scores, dtype=float)


def tracks_to_index_matrix(
    tracks: list[TrackInput],
    session_sizes: SessionSizesInput | None = None,
    *,
    fill_value: int = -1,
):
    """Convert tracks to a dense ``track x session`` ROI-index matrix."""
    assert __backend_name__ != "jax", "Not supported on JAX backend"

    inferred_sizes = _normalize_session_sizes(session_sizes)
    max_session_index = max(inferred_sizes, default=-1)

    for track in tracks:
        seen_sessions: set[int] = set()
        for session_index, detection_index in _iter_track_items(track):
            if detection_index < 0:
                raise ValueError(
                    f"Detection indices must be non-negative, got {detection_index}."
                )
            if session_index in seen_sessions:
                raise ValueError("Each track can only contain one detection per session.")
            seen_sessions.add(session_index)
            max_session_index = max(max_session_index, int(session_index))
            candidate_size = int(detection_index) + 1
            current_size = inferred_sizes.get(int(session_index), 0)
            if session_sizes is None:
                inferred_sizes[int(session_index)] = max(current_size, candidate_size)
            elif candidate_size > current_size:
                raise ValueError(
                    f"Track references detection {detection_index} in session {session_index}, "
                    f"but session_sizes only allows {current_size} detections."
                )

    matrix = full((len(tracks), max_session_index + 1), fill_value, dtype=int)
    for track_index, track in enumerate(tracks):
        for session_index, detection_index in _iter_track_items(track):
            matrix[track_index, session_index] = detection_index
    return matrix


def solve_multisession_assignment_from_similarity(
    pairwise_scores: PairwiseCostsInput,
    session_sizes: SessionSizesInput | None = None,
    *,
    min_score: float | None = None,
    max_gap: int | None = None,
    gap_penalty: float = 0.0,
    start_cost: float = 0.0,
    end_cost: float = 0.0,
    score_to_cost: Callable[[np.ndarray], np.ndarray] | None = None,
) -> MultiSessionAssignmentResult:
    """Score-native wrapper around :func:`solve_multisession_assignment`."""
    assert __backend_name__ != "jax", "Not supported on JAX backend"

    if min_score is not None and not math.isfinite(min_score):
        raise ValueError("min_score must be finite.")
    if max_gap is not None:
        original_max_gap = max_gap
        max_gap = int(max_gap)
        if max_gap != original_max_gap:
            raise ValueError("max_gap must be an integer.")
        if max_gap < 0:
            raise ValueError("max_gap must be non-negative.")

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

    transformed_pairwise_costs: dict[tuple[int, int], np.ndarray] = {}
    for (source_session, target_session), score_matrix in normalized_pairwise_scores.items():
        gap = session_positions[target_session] - session_positions[source_session] - 1
        if max_gap is not None and gap > max_gap:
            continue

        score_matrix_array = np.asarray(score_matrix, dtype=float)
        finite_mask = np.isfinite(score_matrix_array)
        if min_score is not None:
            finite_mask &= score_matrix_array >= float(min_score)

        safe_scores = np.where(np.isfinite(score_matrix_array), score_matrix_array, 0.0)
        cost_matrix = np.asarray(score_to_cost(safe_scores), dtype=float)
        if cost_matrix.shape != score_matrix_array.shape:
            raise ValueError("score_to_cost must preserve the input matrix shape.")

        cost_matrix = cost_matrix.copy()
        cost_matrix[~finite_mask] = np.inf
        if not np.all(np.isfinite(cost_matrix[finite_mask])):
            raise ValueError(
                "score_to_cost must return finite costs for admissible scores."
            )
        transformed_pairwise_costs[(source_session, target_session)] = cost_matrix

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


def _result_to_index_matrix(self, session_sizes: SessionSizesInput | None = None, *, fill_value: int = -1):
    return tracks_to_index_matrix(self.tracks, session_sizes=session_sizes, fill_value=fill_value)


MultiSessionAssignmentResult.to_index_matrix = _result_to_index_matrix  # type: ignore[attr-defined]
