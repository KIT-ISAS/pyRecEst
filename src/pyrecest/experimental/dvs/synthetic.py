"""Synthetic event-count models for DVS active-contour experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .active_contour import activity_profile, rectangle_contour_samples

EDGE_ORDER = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class RectangleCountSimulation:
    """Synthetic rectangle event counts and model probabilities."""

    velocity: np.ndarray
    observed_counts: dict[str, int]
    true_probabilities: dict[str, float]
    normal_flow_probabilities: dict[str, float]
    uniform_probabilities: dict[str, float]


def _as_scalar(value: object, message: str) -> object:
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(message)
    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_, str, bytes, bytearray)):
        raise ValueError(message)
    return scalar


def _validate_positive_finite(value: object, name: str) -> float:
    message = f"{name} must be finite and positive"
    scalar = _as_scalar(value, message)
    try:
        result = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(message)
    return result


def _validate_nonnegative_finite(value: object, name: str) -> float:
    message = f"{name} must be finite and non-negative"
    scalar = _as_scalar(value, message)
    try:
        result = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(message)
    return result


def _validate_nonnegative_integer_count(value: object, name: str) -> int:
    message = f"{name} must be a non-negative integer count"
    scalar = _as_scalar(value, message)
    if isinstance(scalar, (int, np.integer)):
        count = int(scalar)
    else:
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(message)
        count = int(numeric)
    if count < 0:
        raise ValueError(message)
    return count


def _validate_positive_integer_count(value: object, name: str) -> int:
    count = _validate_nonnegative_integer_count(value, name)
    if count <= 0:
        raise ValueError(f"{name} must be a positive integer count")
    return count


def summarize_edge_counts(
    edge_labels: list[str], point_counts: np.ndarray
) -> dict[str, int]:
    """Aggregate per-contour-sample counts by edge label."""
    labels = np.array(edge_labels)
    return {edge: int(np.sum(point_counts[labels == edge])) for edge in EDGE_ORDER}


def _edge_probabilities(
    edge_labels: list[str], point_weights: np.ndarray
) -> dict[str, float]:
    labels = np.array(edge_labels)
    weights = np.asarray(point_weights, dtype=float)
    if weights.shape != labels.shape:
        raise ValueError("point_weights must contain one value per edge label")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("point_weights must contain only finite non-negative values")
    edge_weights = np.array(
        [float(np.sum(weights[labels == edge])) for edge in EDGE_ORDER],
        dtype=float,
    )
    total_weight = float(np.sum(edge_weights))
    if total_weight <= 0.0:
        edge_weights = np.ones(len(EDGE_ORDER), dtype=float)
        total_weight = float(len(EDGE_ORDER))
    return {
        edge: float(weight / total_weight)
        for edge, weight in zip(EDGE_ORDER, edge_weights, strict=True)
    }


def edge_probabilities_from_activity(
    edge_labels: list[str],
    activities: np.ndarray,
    background_activity: float = 1e-3,
) -> dict[str, float]:
    """Convert normal-flow activities into edge-level event probabilities."""
    background_activity = _validate_nonnegative_finite(
        background_activity, "background_activity"
    )
    weights = np.asarray(activities, dtype=float) + background_activity
    return _edge_probabilities(edge_labels, weights)


def uniform_edge_probabilities(edge_labels: list[str]) -> dict[str, float]:
    """Return edge probabilities under a motion-blind uniform contour model."""
    return _edge_probabilities(edge_labels, np.ones(len(edge_labels), dtype=float))


def count_negative_log_likelihood(
    observed_counts: dict[str, int],
    probabilities: dict[str, float],
    probability_floor: float = 1e-12,
) -> float:
    """Return multinomial count NLL up to the count-dependent constant."""
    probability_floor = _validate_positive_finite(
        probability_floor, "probability_floor"
    )
    nll = 0.0
    for edge in EDGE_ORDER:
        count = _validate_nonnegative_integer_count(
            observed_counts[edge], f"observed_counts[{edge!r}]"
        )
        probability = _validate_nonnegative_finite(
            probabilities[edge], f"probabilities[{edge!r}]"
        )
        nll -= count * float(np.log(max(probability, probability_floor)))
    return nll


def simulate_rectangle_event_counts(
    velocity: np.ndarray,
    total_events: int = 240,
    width: float = 2.0,
    height: float = 1.0,
    samples_per_edge: int = 80,
    background_activity: float = 1e-3,
    seed: int | None = 0,
) -> RectangleCountSimulation:
    """Sample event counts from a motion-gated rectangle contour model."""
    total_events = _validate_positive_integer_count(total_events, "total_events")
    background_activity = _validate_nonnegative_finite(
        background_activity, "background_activity"
    )

    contour = rectangle_contour_samples(
        width=width,
        height=height,
        samples_per_edge=samples_per_edge,
    )
    activities = activity_profile(contour.normals, velocity)
    point_weights = activities + background_activity
    point_weight_sum = float(np.sum(point_weights))
    if not np.isfinite(point_weight_sum) or point_weight_sum <= 0.0:
        raise ValueError("event-generation weights must have positive finite sum")
    point_probabilities = point_weights / point_weight_sum
    rng = np.random.default_rng(seed)
    point_counts = rng.multinomial(total_events, point_probabilities)

    true_probabilities = edge_probabilities_from_activity(
        contour.edge_labels,
        activities,
        background_activity=background_activity,
    )
    return RectangleCountSimulation(
        velocity=np.asarray(velocity, dtype=float),
        observed_counts=summarize_edge_counts(contour.edge_labels, point_counts),
        true_probabilities=true_probabilities,
        normal_flow_probabilities=true_probabilities,
        uniform_probabilities=uniform_edge_probabilities(contour.edge_labels),
    )
