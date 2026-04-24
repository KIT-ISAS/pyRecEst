"""Shared helpers for multi-session assignment utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pyrecest.backend import asarray  # pylint: disable=no-name-in-module

Observation = tuple[int, int]
PairwiseCostsInput = Mapping[tuple[int, int], Any] | Sequence[Any]
SessionSizesInput = Mapping[int, int] | Sequence[int]
TrackInput = Mapping[int, int] | Sequence[Observation]
BackendArray = Any


def normalize_pairwise_costs(
    pairwise_costs: PairwiseCostsInput,
) -> dict[tuple[int, int], BackendArray]:
    """Normalize pairwise costs to a mapping keyed by session pair."""
    if isinstance(pairwise_costs, Mapping):
        normalized: dict[tuple[int, int], BackendArray] = {}
        for key, value in pairwise_costs.items():
            if len(key) != 2:
                raise ValueError("Each pairwise-cost key must contain two session indices.")
            source_session, target_session = int(key[0]), int(key[1])
            if source_session >= target_session:
                raise ValueError("Pairwise-cost keys must satisfy source_session < target_session.")
            matrix = asarray(value, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("Each pairwise cost matrix must be two-dimensional.")
            normalized[(source_session, target_session)] = matrix
        return normalized

    normalized = {}
    for session_idx, value in enumerate(pairwise_costs):
        matrix = asarray(value, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("Each pairwise cost matrix must be two-dimensional.")
        normalized[(session_idx, session_idx + 1)] = matrix
    return normalized


def normalize_session_sizes(
    session_sizes: SessionSizesInput | None,
) -> dict[int, int]:
    """Normalize session sizes to a mapping keyed by session index."""
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


def infer_and_validate_session_sizes(
    pairwise_costs: Mapping[tuple[int, int], BackendArray],
    session_sizes: Mapping[int, int],
) -> dict[int, int]:
    """Infer session sizes from pairwise matrices and validate explicit sizes."""
    inferred_sizes = dict(session_sizes)
    for (source_session, target_session), cost_matrix in pairwise_costs.items():
        source_size, target_size = cost_matrix.shape
        _check_or_set_session_size(inferred_sizes, source_session, source_size)
        _check_or_set_session_size(inferred_sizes, target_session, target_size)

    if not inferred_sizes and not pairwise_costs:
        raise ValueError("No observations were provided. Supply pairwise_costs or session_sizes.")

    return dict(sorted(inferred_sizes.items()))


def _check_or_set_session_size(
    inferred_sizes: dict[int, int],
    session_idx: int,
    candidate_size: int,
) -> None:
    if session_idx in inferred_sizes and inferred_sizes[session_idx] != candidate_size:
        raise ValueError(
            f"Inconsistent detection count for session {session_idx}: "
            f"expected {inferred_sizes[session_idx]}, got {candidate_size}."
        )
    inferred_sizes[session_idx] = int(candidate_size)


def build_session_offsets(session_sizes: Mapping[int, int]) -> dict[int, int]:
    """Return the first global observation index for each session."""
    offsets: dict[int, int] = {}
    offset = 0
    for session_idx in sorted(session_sizes):
        offsets[session_idx] = offset
        offset += int(session_sizes[session_idx])
    return offsets


def build_observation_index(
    session_sizes: Mapping[int, int],
) -> tuple[list[Observation], dict[Observation, int]]:
    """Build forward and reverse mappings between global and session indices."""
    global_to_observation: list[Observation] = []
    observation_to_global: dict[Observation, int] = {}

    for session_idx in sorted(session_sizes):
        for detection_idx in range(session_sizes[session_idx]):
            observation = (session_idx, detection_idx)
            observation_to_global[observation] = len(global_to_observation)
            global_to_observation.append(observation)

    return global_to_observation, observation_to_global


def session_gap(
    source_session: int,
    target_session: int,
    session_positions: Mapping[int, int],
) -> int:
    """Return skipped-session count for a forward session edge."""
    source_position = session_positions[source_session]
    target_position = session_positions[target_session]
    gap = target_position - source_position - 1
    if gap < 0:
        raise ValueError("Session indices must define a forward-in-time edge ordering.")
    return gap


def iter_track_items(track: TrackInput) -> list[Observation]:
    """Return sorted ``(session, detection)`` pairs from any supported track input."""
    if isinstance(track, Mapping):
        items = list(track.items())
    else:
        items = [(int(session_idx), int(detection_idx)) for session_idx, detection_idx in track]
    items.sort(key=lambda item: item[0])
    return [(int(session_idx), int(detection_idx)) for session_idx, detection_idx in items]


def infer_track_session_sizes(
    tracks: Sequence[TrackInput],
    session_sizes: SessionSizesInput | None,
    *,
    require_unique_sessions: bool = False,
) -> tuple[dict[int, int], int]:
    """Infer session sizes referenced by tracks and validate explicit bounds."""
    inferred_sizes = normalize_session_sizes(session_sizes)
    max_session_index = max(inferred_sizes, default=-1)

    for track in tracks:
        seen_sessions: set[int] = set()
        for session_index, detection_index in iter_track_items(track):
            if detection_index < 0:
                raise ValueError(
                    f"Detection indices must be non-negative, got {detection_index}."
                )

            session_index = int(session_index)
            if require_unique_sessions and session_index in seen_sessions:
                raise ValueError("Each track can only contain one detection per session.")
            seen_sessions.add(session_index)

            max_session_index = max(max_session_index, session_index)
            candidate_size = int(detection_index) + 1
            current_size = inferred_sizes.get(session_index, 0)
            if session_sizes is None:
                inferred_sizes[session_index] = max(current_size, candidate_size)
            elif candidate_size > current_size:
                raise ValueError(
                    f"Track references detection {detection_index} in session {session_index}, "
                    f"but session_sizes only allows {current_size} detections."
                )

    return inferred_sizes, max_session_index
