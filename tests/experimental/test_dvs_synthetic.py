import numpy as np
import pytest
from pyrecest.experimental.dvs import (
    count_negative_log_likelihood,
    simulate_rectangle_event_counts,
    uniform_edge_probabilities,
)

_EDGES = ("left", "right", "top", "bottom")


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


def test_count_nll_rejects_invalid_observed_counts():
    probabilities = {edge: 0.25 for edge in _EDGES}

    for bad_count in (-1, 1.5, np.nan, np.inf, True, "2", np.array([2])):
        observed_counts = {edge: 1 for edge in _EDGES}
        observed_counts["left"] = bad_count
        with pytest.raises(ValueError, match="observed_counts"):
            count_negative_log_likelihood(observed_counts, probabilities)


def test_count_nll_rejects_invalid_probabilities():
    observed_counts = {edge: 1 for edge in _EDGES}

    for bad_probability in (-0.1, 1.1, np.nan, np.inf, True, "0.25", 0.25 + 0.1j):
        probabilities = {edge: 0.25 for edge in _EDGES}
        probabilities["left"] = bad_probability
        with pytest.raises(ValueError, match="probabilities"):
            count_negative_log_likelihood(observed_counts, probabilities)


def test_count_nll_rejects_invalid_probability_floor():
    observed_counts = {edge: 1 for edge in _EDGES}
    probabilities = {edge: 0.25 for edge in _EDGES}

    for probability_floor in (0.0, -1e-3, 1.1, np.nan, np.inf, True, "1e-12"):
        with pytest.raises(ValueError, match="probability_floor"):
            count_negative_log_likelihood(
                observed_counts,
                probabilities,
                probability_floor=probability_floor,
            )
