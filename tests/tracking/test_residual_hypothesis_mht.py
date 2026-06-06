from __future__ import annotations

from pyrecest.tracking import (
    ResidualEditCandidate,
    ResidualMHTConfig,
    enumerate_residual_hypotheses,
    select_residual_hypothesis,
)


def test_selects_best_compatible_residual_hypothesis():
    candidates = [
        ResidualEditCandidate("a", 2.0, frozenset({"target:1"})),
        ResidualEditCandidate("b", 1.8, frozenset({"target:2"})),
        ResidualEditCandidate("c", 3.0, frozenset({"target:1"})),
    ]

    selected = select_residual_hypothesis(
        candidates,
        config=ResidualMHTConfig(max_edits=2, edit_penalty=0.1, score_threshold=0.0),
    )

    assert selected.candidate_ids == ("c", "b")
    assert selected.n_edits == 2


def test_returns_no_edit_below_threshold():
    selected = select_residual_hypothesis(
        [ResidualEditCandidate("weak", 0.5)],
        config=ResidualMHTConfig(max_edits=1, edit_penalty=0.0, score_threshold=1.0),
    )

    assert selected.candidate_ids == ()
    assert selected.score == 0.0


def test_enumerates_no_edit_when_requested():
    hypotheses = enumerate_residual_hypotheses(
        [ResidualEditCandidate("a", 1.0)],
        config=ResidualMHTConfig(max_edits=1, include_empty=True),
    )

    assert any(hypothesis.candidate_ids == () for hypothesis in hypotheses)


def test_conflicting_candidates_are_not_combined():
    hypotheses = enumerate_residual_hypotheses(
        [
            ResidualEditCandidate("a", 1.0, frozenset({"x"})),
            ResidualEditCandidate("b", 1.0, frozenset({"x"})),
        ],
        config=ResidualMHTConfig(max_edits=2, include_empty=False),
    )

    assert all(len(hypothesis.candidate_ids) == 1 for hypothesis in hypotheses)
