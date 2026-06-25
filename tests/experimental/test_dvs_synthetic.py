import numpy as np
import pytest
from pyrecest.experimental.dvs import (
    EDGE_ORDER,
    count_negative_log_likelihood,
    edge_probabilities_from_activity,
    simulate_rectangle_event_counts,
    uniform_edge_probabilities,
)


def _counts(value=1):
    return {edge: value for edge in EDGE_ORDER}


def _probabilities(value=0.25):
    return {edge: value for edge in EDGE_ORDER}


def test_motion_gated_counts_concentrate_on_vertical_edges():
    simulation = simulate_rectangle_event_counts(
        np.array([1.0, 0.0]),
        total_events=200,
        background_activity=0.0,
        seed=1,
    )

    assert (
        simulation.observed_counts["left"] + simulation.observed_counts["right"] == 200
    )
    assert simulation.observed_counts["top"] + simulation.observed_counts["bottom"] == 0


def test_normal_flow_likelihood_beats_uniform_for_horizontal_motion():
    simulation = simulate_rectangle_event_counts(
        np.array([1.0, 0.0]),
        total_events=200,
        background_activity=1e-3,
        seed=2,
    )

    normal_flow_nll = count_negative_log_likelihood(
        simulation.observed_counts, simulation.normal_flow_probabilities
    )
    uniform_nll = count_negative_log_likelihood(
        simulation.observed_counts, simulation.uniform_probabilities
    )

    assert normal_flow_nll < uniform_nll


def test_uniform_edge_probabilities_sum_to_one():
    probabilities = uniform_edge_probabilities(
        ["left", "left", "right", "right", "top", "bottom"]
    )

    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_count_negative_log_likelihood_uses_probability_floor_for_zero_probability():
    counts = _counts(0)
    counts["left"] = 2
    probabilities = _probabilities(0.25)
    probabilities["left"] = 0.0

    nll = count_negative_log_likelihood(counts, probabilities, probability_floor=1e-3)

    assert nll == pytest.approx(-2.0 * np.log(1e-3))


@pytest.mark.parametrize(
    "bad_count",
    [-1, 1.5, np.nan, np.inf, True, "2", np.array([1])],
)
def test_count_negative_log_likelihood_rejects_invalid_counts(bad_count):
    counts = _counts(1)
    counts["left"] = bad_count

    with pytest.raises(ValueError, match="observed_counts"):
        count_negative_log_likelihood(counts, _probabilities())


@pytest.mark.parametrize(
    "bad_probability",
    [-0.1, np.nan, np.inf, True, "0.25", np.array([0.25])],
)
def test_count_negative_log_likelihood_rejects_invalid_probabilities(bad_probability):
    probabilities = _probabilities(0.25)
    probabilities["left"] = bad_probability

    with pytest.raises(ValueError, match="probabilities"):
        count_negative_log_likelihood(_counts(), probabilities)


@pytest.mark.parametrize(
    "bad_floor",
    [0.0, -1.0, np.nan, np.inf, True, "1e-12", np.array([1e-12])],
)
def test_count_negative_log_likelihood_rejects_invalid_probability_floor(bad_floor):
    with pytest.raises(ValueError, match="probability_floor"):
        count_negative_log_likelihood(
            _counts(), _probabilities(), probability_floor=bad_floor
        )


@pytest.mark.parametrize(
    "bad_background",
    [np.nan, np.inf, -1.0, True, "0.1", np.array([0.1])],
)
def test_edge_probabilities_from_activity_rejects_invalid_background_activity(
    bad_background,
):
    edge_labels = list(EDGE_ORDER)
    activities = np.ones(len(edge_labels), dtype=float)

    with pytest.raises(ValueError, match="background_activity"):
        edge_probabilities_from_activity(
            edge_labels,
            activities,
            background_activity=bad_background,
        )


@pytest.mark.parametrize(
    "bad_activities",
    [
        [np.nan, 1.0, 1.0, 1.0],
        [np.inf, 1.0, 1.0, 1.0],
        [-2.0, 1.0, 1.0, 1.0],
    ],
)
def test_edge_probabilities_from_activity_rejects_invalid_weights(bad_activities):
    with pytest.raises(ValueError, match="point_weights"):
        edge_probabilities_from_activity(list(EDGE_ORDER), np.array(bad_activities))


@pytest.mark.parametrize("bad_total_events", [0, -1, 1.5, np.nan, np.inf, True, "4"])
def test_simulate_rectangle_event_counts_rejects_invalid_total_events(bad_total_events):
    with pytest.raises(ValueError, match="total_events"):
        simulate_rectangle_event_counts(
            np.array([1.0, 0.0]),
            total_events=bad_total_events,
            samples_per_edge=2,
        )
