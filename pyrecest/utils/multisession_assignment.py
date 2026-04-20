"""Global multi-session association utilities.

This module solves a common batch-linking problem for longitudinal cell tracking:
start from pairwise ROI matching costs between imaging sessions and recover a set
of globally consistent tracks, with at most one detection per session on each
track. The formulation supports skipped sessions through optional cross-session
edges and a configurable gap penalty.

The optimization is implemented as a minimum-cost path-cover problem on a DAG and
reduced to a single linear-sum assignment problem. This makes it a good fit for
cross-session calcium imaging where pairwise costs are already available from
registration, centroid distances, ROI overlap, or other similarity cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

Observation = Tuple[int, int]
MatchedEdge = Tuple[Observation, Observation, float]
PairwiseCostsInput = Union[
    Mapping[Tuple[int, int], np.ndarray],
    Sequence[np.ndarray],
]
SessionSizesInput = Union[Mapping[int, int], Sequence[int]]


@dataclass(frozen=True)
class MultiSessionAssignmentResult:
    """Result of :func:`solve_multisession_assignment`.

    Attributes
    ----------
    tracks
        One dictionary per recovered track. Keys are session indices and values
        are detection indices within that session.
    matched_edges
        Directed edges selected by the optimizer as
        ``((source_session, source_detection), (target_session, target_detection), cost)``.
    total_cost
        Objective value of the globally optimal assignment, including track start
        and end costs.
    """

    tracks: List[Dict[int, int]]
    matched_edges: List[MatchedEdge]
    total_cost: float

    def observation_to_track_index(self) -> Dict[Observation, int]:
        """Return a mapping from observation to recovered track index."""
        mapping: Dict[Observation, int] = {}
        for track_idx, track in enumerate(self.tracks):
            for session_idx, detection_idx in track.items():
                mapping[(session_idx, detection_idx)] = track_idx
        return mapping


def solve_multisession_assignment(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    pairwise_costs: PairwiseCostsInput,
    session_sizes: Optional[SessionSizesInput] = None,
    *,
    start_cost: float = 0.0,
    end_cost: float = 0.0,
    gap_penalty: float = 0.0,
    cost_threshold: Optional[float] = None,
) -> MultiSessionAssignmentResult:
    """Recover globally consistent tracks across multiple sessions.

    Parameters
    ----------
    pairwise_costs
        Either a mapping from ``(source_session, target_session)`` to a cost
        matrix of shape ``(n_source, n_target)``, or a sequence of consecutive
        session-to-session cost matrices. When a sequence is supplied, matrix
        ``k`` is interpreted as the cost matrix for session ``k`` to session
        ``k + 1``.
    session_sizes
        Optional per-session detection counts. This is only required when some
        sessions have no pairwise cost matrix, or when isolated singleton
        detections should still be represented in the result. A sequence is
        interpreted as counts for sessions ``0, 1, ..., S-1``.
    start_cost
        Penalty for starting a new track.
    end_cost
        Penalty for ending a track.
    gap_penalty
        Additional penalty applied per skipped session for non-consecutive edges.
        This penalty is added on top of the supplied edge cost.
    cost_threshold
        Optional upper bound for admissible link costs after gap penalties are
        applied. Edges with larger costs are forbidden.

    Returns
    -------
    MultiSessionAssignmentResult
        Globally optimal tracks and the selected directed edges.

    Notes
    -----
    The objective is a weighted path cover on a directed acyclic graph where each
    detection can have at most one predecessor and one successor. This is a good
    model for longitudinal neuron identity tracking across discrete imaging
    sessions because a neuron should appear at most once per session, while
    temporary missed detections can be handled by adding cross-session edges.
    """

    normalized_pairwise_costs = _normalize_pairwise_costs(pairwise_costs)
    normalized_session_sizes = _normalize_session_sizes(session_sizes)
    session_sizes_map = _infer_and_validate_session_sizes(
        normalized_pairwise_costs,
        normalized_session_sizes,
    )

    if not session_sizes_map:
        return MultiSessionAssignmentResult(tracks=[], matched_edges=[], total_cost=0.0)

    session_order = sorted(session_sizes_map)
    session_positions = {session_idx: pos for pos, session_idx in enumerate(session_order)}
    global_to_observation, observation_to_global = _build_observation_index(session_sizes_map)
    n_observations = len(global_to_observation)

    if n_observations == 0:
        return MultiSessionAssignmentResult(tracks=[], matched_edges=[], total_cost=0.0)

    all_finite_costs = _collect_finite_costs(
        normalized_pairwise_costs.values(),
        [start_cost, end_cost, gap_penalty],
    )
    invalid_cost = _compute_invalid_cost(all_finite_costs, n_observations)

    assignment_costs = np.full(
        (2 * n_observations, 2 * n_observations),
        invalid_cost,
        dtype=float,
    )
    assignment_costs[n_observations:, n_observations:] = 0.0

    for observation_idx in range(n_observations):
        assignment_costs[observation_idx, n_observations + observation_idx] = float(end_cost)
        assignment_costs[n_observations + observation_idx, observation_idx] = float(start_cost)

    edge_costs: Dict[Tuple[int, int], float] = {}
    for (source_session, target_session), source_target_costs in normalized_pairwise_costs.items():
        source_position = session_positions[source_session]
        target_position = session_positions[target_session]
        gap = target_position - source_position - 1
        if gap < 0:
            raise ValueError(
                "Session indices must define a forward-in-time edge ordering."
            )

        for source_detection in range(source_target_costs.shape[0]):
            source_global = observation_to_global[(source_session, source_detection)]
            for target_detection in range(source_target_costs.shape[1]):
                raw_cost = float(source_target_costs[source_detection, target_detection])
                if not np.isfinite(raw_cost):
                    continue
                link_cost = raw_cost + gap_penalty * gap
                if cost_threshold is not None and link_cost > cost_threshold:
                    continue

                target_global = observation_to_global[(target_session, target_detection)]
                current_cost = assignment_costs[source_global, target_global]
                if link_cost < current_cost:
                    assignment_costs[source_global, target_global] = link_cost
                    edge_costs[(source_global, target_global)] = link_cost

    row_indices, col_indices = linear_sum_assignment(assignment_costs)
    row_to_col = np.full(2 * n_observations, -1, dtype=int)
    row_to_col[row_indices] = col_indices
    total_cost = float(assignment_costs[row_indices, col_indices].sum())

    successors: Dict[int, Optional[int]] = {idx: None for idx in range(n_observations)}
    predecessors: Dict[int, Optional[int]] = {idx: None for idx in range(n_observations)}
    matched_edges: List[MatchedEdge] = []

    for source_global in range(n_observations):
        assigned_column = int(row_to_col[source_global])
        if assigned_column < 0 or assigned_column >= n_observations:
            continue
        if assignment_costs[source_global, assigned_column] >= invalid_cost:
            continue
        successors[source_global] = assigned_column
        predecessors[assigned_column] = source_global
        matched_edges.append(
            (
                global_to_observation[source_global],
                global_to_observation[assigned_column],
                edge_costs[(source_global, assigned_column)],
            )
        )

    tracks = _reconstruct_tracks(
        global_to_observation,
        predecessors,
        successors,
        session_positions,
    )

    matched_edges.sort(
        key=lambda edge: (
            session_positions[edge[0][0]],
            edge[0][1],
            session_positions[edge[1][0]],
            edge[1][1],
        )
    )

    return MultiSessionAssignmentResult(
        tracks=tracks,
        matched_edges=matched_edges,
        total_cost=total_cost,
    )


def _normalize_pairwise_costs(
    pairwise_costs: PairwiseCostsInput,
) -> Dict[Tuple[int, int], np.ndarray]:
    if isinstance(pairwise_costs, Mapping):
        normalized: Dict[Tuple[int, int], np.ndarray] = {}
        for key, value in pairwise_costs.items():
            if len(key) != 2:
                raise ValueError("Each pairwise-cost key must contain two session indices.")
            source_session, target_session = int(key[0]), int(key[1])
            if source_session >= target_session:
                raise ValueError("Pairwise-cost keys must satisfy source_session < target_session.")
            matrix = np.asarray(value, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("Each pairwise cost matrix must be two-dimensional.")
            normalized[(source_session, target_session)] = matrix
        return normalized

    normalized = {}
    for session_idx, value in enumerate(pairwise_costs):
        matrix = np.asarray(value, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Each pairwise cost matrix must be two-dimensional.")
        normalized[(session_idx, session_idx + 1)] = matrix
    return normalized


def _normalize_session_sizes(
    session_sizes: Optional[SessionSizesInput],
) -> Dict[int, int]:
    if session_sizes is None:
        return {}
    if isinstance(session_sizes, Mapping):
        normalized = {int(session_idx): int(size) for session_idx, size in session_sizes.items()}
    else:
        normalized = {session_idx: int(size) for session_idx, size in enumerate(session_sizes)}
    for session_idx, size in normalized.items():
        if size < 0:
            raise ValueError(f"Session {session_idx} has a negative detection count.")
    return normalized


def _infer_and_validate_session_sizes(
    pairwise_costs: Mapping[Tuple[int, int], np.ndarray],
    session_sizes: Mapping[int, int],
) -> Dict[int, int]:
    inferred_sizes = dict(session_sizes)
    for (source_session, target_session), cost_matrix in pairwise_costs.items():
        source_size, target_size = cost_matrix.shape
        _check_or_set_session_size(inferred_sizes, source_session, source_size)
        _check_or_set_session_size(inferred_sizes, target_session, target_size)

    if not inferred_sizes and not pairwise_costs:
        raise ValueError(
            "No observations were provided. Supply pairwise_costs or session_sizes."
        )

    return dict(sorted(inferred_sizes.items()))


def _check_or_set_session_size(
    inferred_sizes: Dict[int, int],
    session_idx: int,
    candidate_size: int,
) -> None:
    if session_idx in inferred_sizes and inferred_sizes[session_idx] != candidate_size:
        raise ValueError(
            f"Inconsistent detection count for session {session_idx}: "
            f"expected {inferred_sizes[session_idx]}, got {candidate_size}."
        )
    inferred_sizes[session_idx] = int(candidate_size)


def _build_observation_index(
    session_sizes: Mapping[int, int],
) -> Tuple[List[Observation], Dict[Observation, int]]:
    global_to_observation: List[Observation] = []
    observation_to_global: Dict[Observation, int] = {}

    for session_idx in sorted(session_sizes):
        for detection_idx in range(session_sizes[session_idx]):
            observation = (session_idx, detection_idx)
            observation_to_global[observation] = len(global_to_observation)
            global_to_observation.append(observation)

    return global_to_observation, observation_to_global


def _collect_finite_costs(
    matrices: Iterable[np.ndarray],
    scalar_costs: Sequence[float],
) -> List[float]:
    finite_costs = [float(cost) for cost in scalar_costs if np.isfinite(cost)]
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=float)
        finite_entries = matrix[np.isfinite(matrix)]
        if finite_entries.size:
            finite_costs.extend(float(entry) for entry in finite_entries.ravel())
    return finite_costs


def _compute_invalid_cost(finite_costs: Sequence[float], n_observations: int) -> float:
    scale = max([1.0] + [abs(cost) for cost in finite_costs])
    return float(scale * max(1000.0, 100.0 * n_observations + 1.0))


def _reconstruct_tracks(
    global_to_observation: Sequence[Observation],
    predecessors: Mapping[int, Optional[int]],
    successors: Mapping[int, Optional[int]],
    session_positions: Mapping[int, int],
) -> List[Dict[int, int]]:
    visited = set()
    tracks: List[Dict[int, int]] = []

    for node_idx in range(len(global_to_observation)):
        if predecessors[node_idx] is not None:
            continue
        if node_idx in visited:
            continue

        track: Dict[int, int] = {}
        current_node: Optional[int] = node_idx
        while current_node is not None:
            if current_node in visited:
                raise RuntimeError("Encountered a cycle while reconstructing tracks.")
            visited.add(current_node)
            session_idx, detection_idx = global_to_observation[current_node]
            track[session_idx] = detection_idx
            current_node = successors[current_node]
        tracks.append(track)

    tracks.sort(
        key=lambda track: (
            min(session_positions[session_idx] for session_idx in track),
            min(track.values()),
            len(track),
        )
    )
    return tracks
