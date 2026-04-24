"""Extensions for multi-session assignment with observation-specific start/end costs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

from pyrecest.backend import asarray, full, isfinite

from .multisession_assignment import (
    MultiSessionAssignmentResult,
    PairwiseCostsInput,
    SessionSizesInput,
    _compute_session_gap,
    _infer_and_validate_session_sizes,
    _normalize_pairwise_costs,
    _normalize_session_sizes,
    solve_multisession_assignment,
)

ObservationCostsInput: TypeAlias = Mapping[int, Any] | Sequence[Any]


def solve_multisession_assignment_with_observation_costs(  # pylint: disable=too-many-arguments,too-many-locals
    pairwise_costs: PairwiseCostsInput,
    session_sizes: SessionSizesInput | None = None,
    *,
    start_cost: float = 0.0,
    end_cost: float = 0.0,
    start_costs: ObservationCostsInput | None = None,
    end_costs: ObservationCostsInput | None = None,
    gap_penalty: float = 0.0,
    cost_threshold: float | None = None,
) -> MultiSessionAssignmentResult:
    """Solve global multi-session assignment with heterogeneous birth/death costs.

    This is a thin wrapper around :func:`solve_multisession_assignment`. It encodes
    observation-specific start/end costs into transformed pairwise costs and then
    maps the reported objective value and matched-edge costs back to the original
    problem.
    """

    normalized_pairwise = _normalize_pairwise_costs(pairwise_costs)
    normalized_sizes = _normalize_session_sizes(session_sizes)
    session_sizes_map = _infer_and_validate_session_sizes(
        normalized_pairwise,
        normalized_sizes,
    )

    if not session_sizes_map:
        return MultiSessionAssignmentResult(tracks=[], matched_edges=[], total_cost=0.0)

    normalized_start_costs = _normalize_observation_costs(
        start_costs,
        session_sizes_map,
        default_value=float(start_cost),
        name="start_costs",
    )
    normalized_end_costs = _normalize_observation_costs(
        end_costs,
        session_sizes_map,
        default_value=float(end_cost),
        name="end_costs",
    )

    max_start_cost = max(
        float(costs.max()) if costs.size else float(start_cost)
        for costs in normalized_start_costs.values()
    )
    max_end_cost = max(
        float(costs.max()) if costs.size else float(end_cost)
        for costs in normalized_end_costs.values()
    )

    session_order = sorted(session_sizes_map)
    session_positions = {
        session_idx: position for position, session_idx in enumerate(session_order)
    }

    transformed_pairwise = _transform_pairwise_costs(
        normalized_pairwise,
        session_positions,
        normalized_start_costs,
        normalized_end_costs,
        uniform_start_cost=max_start_cost,
        uniform_end_cost=max_end_cost,
        gap_penalty=float(gap_penalty),
        cost_threshold=cost_threshold,
    )

    base_result = solve_multisession_assignment(
        transformed_pairwise,
        session_sizes=session_sizes_map,
        start_cost=max_start_cost,
        end_cost=max_end_cost,
        gap_penalty=gap_penalty,
        cost_threshold=None,
    )

    actual_baseline = sum(
        float(costs.sum()) for costs in normalized_start_costs.values()
    ) + sum(float(costs.sum()) for costs in normalized_end_costs.values())
    uniform_baseline = sum(session_sizes_map.values()) * (max_start_cost + max_end_cost)
    total_cost = float(base_result.total_cost - uniform_baseline + actual_baseline)

    adjusted_edges = [
        (
            source,
            target,
            float(
                transformed_cost
                - max_end_cost
                - max_start_cost
                + normalized_end_costs[source[0]][source[1]]
                + normalized_start_costs[target[0]][target[1]]
            ),
        )
        for source, target, transformed_cost in base_result.matched_edges
    ]

    return MultiSessionAssignmentResult(
        tracks=base_result.tracks,
        matched_edges=adjusted_edges,
        total_cost=total_cost,
    )


def _normalize_observation_costs(
    observation_costs: ObservationCostsInput | None,
    session_sizes: Mapping[int, int],
    *,
    default_value: float,
    name: str,
) -> dict[int, Any]:
    if observation_costs is None:
        raw_entries: dict[int, Any] = {}
    elif isinstance(observation_costs, Mapping):
        raw_entries = {int(session_idx): value for session_idx, value in observation_costs.items()}
    else:
        raw_entries = dict(enumerate(observation_costs))

    unknown_sessions = sorted(set(raw_entries) - set(session_sizes))
    if unknown_sessions:
        raise ValueError(
            f"{name} specifies unknown sessions {unknown_sessions}; "
            f"known sessions are {sorted(session_sizes)}."
        )

    normalized: dict[int, Any] = {}
    for session_idx in sorted(session_sizes):
        session_size = int(session_sizes[session_idx])
        if session_idx not in raw_entries:
            values = full(session_size, float(default_value), dtype=float)
        else:
            values = _normalize_observation_cost_entry(
                raw_entries[session_idx],
                session_size,
                session_idx=session_idx,
                name=name,
            )
        if not bool(isfinite(values).all()):
            raise ValueError(f"{name} for session {session_idx} must contain only finite values.")
        normalized[session_idx] = values
    return normalized


def _normalize_observation_cost_entry(
    value: Any,
    session_size: int,
    *,
    session_idx: int,
    name: str,
) -> Any:
    values = asarray(value, dtype=float)
    if values.ndim == 0:
        return full(session_size, float(values), dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"{name} entry for session {session_idx} must be a scalar or a one-dimensional array."
        )
    if int(values.size) != int(session_size):
        raise ValueError(
            f"{name} entry for session {session_idx} has length {values.size}, expected {session_size}."
        )
    return values


def _transform_pairwise_costs(  # pylint: disable=too-many-arguments,too-many-locals
    pairwise_costs: Mapping[tuple[int, int], Any],
    session_positions: Mapping[int, int],
    start_costs: Mapping[int, Any],
    end_costs: Mapping[int, Any],
    *,
    uniform_start_cost: float,
    uniform_end_cost: float,
    gap_penalty: float,
    cost_threshold: float | None,
) -> dict[tuple[int, int], Any]:
    transformed: dict[tuple[int, int], Any] = {}
    for (source_session, target_session), matrix in pairwise_costs.items():
        gap = _compute_session_gap(
            session_positions,
            source_session,
            target_session,
        )
        matrix_array = asarray(matrix, dtype=float)
        source_end = end_costs[source_session][:, None]
        target_start = start_costs[target_session][None, :]
        transformed_matrix = (
            matrix_array
            - source_end
            - target_start
            + float(uniform_end_cost)
            + float(uniform_start_cost)
        )

        if cost_threshold is not None:
            adjusted_original = matrix_array + float(gap_penalty) * gap
            transformed_matrix = transformed_matrix.copy()
            transformed_matrix[adjusted_original > float(cost_threshold)] = math.inf

        transformed[(source_session, target_session)] = transformed_matrix
    return transformed


__all__ = ["solve_multisession_assignment_with_observation_costs"]
