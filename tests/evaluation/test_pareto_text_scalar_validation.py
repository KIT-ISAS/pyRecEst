from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pyrecest.evaluation import (
    constraint_mask,
    equal_quality_selection,
    is_pareto_front,
    pareto_front_indices,
    record_dominates,
    select_under_constraints,
)


def _small_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "a", "error": 1.0, "runtime": 2.0},
            {"name": "b", "error": 2.0, "runtime": 1.0},
        ]
    )


@pytest.mark.parametrize("eps", ["1e-12", b"1e-12", np.array("1e-12")])
def test_pareto_helpers_reject_text_scalar_eps(eps: object) -> None:
    table = _small_table()

    with pytest.raises(ValueError, match="eps"):
        pareto_front_indices(
            table,
            ["error", "runtime"],
            directions={"error": "min", "runtime": "min"},
            eps=eps,
        )
    with pytest.raises(ValueError, match="eps"):
        is_pareto_front(
            table,
            ["error", "runtime"],
            directions={"error": "min", "runtime": "min"},
            eps=eps,
        )
    with pytest.raises(ValueError, match="eps"):
        record_dominates(
            table.iloc[0],
            table.iloc[1],
            ["error", "runtime"],
            directions={"error": "min", "runtime": "min"},
            eps=eps,
        )
    with pytest.raises(ValueError, match="eps"):
        constraint_mask(table, {"error": ("<=", 2.0)}, eps=eps)
    with pytest.raises(ValueError, match="eps"):
        select_under_constraints(
            table,
            constraints={"error": ("<=", 2.0)},
            objective="runtime",
            direction="min",
            eps=eps,
        )
    with pytest.raises(ValueError, match="eps"):
        equal_quality_selection(
            table,
            quality_constraints={"error": ("<=", 2.0)},
            compression_objective="runtime",
            eps=eps,
        )


@pytest.mark.parametrize("threshold", ["1.0", b"1.0", np.array("1.0")])
def test_pareto_constraint_thresholds_reject_text_scalars(threshold: object) -> None:
    table = pd.DataFrame([{"score": 1.0}])

    with pytest.raises(
        ValueError,
        match="Constraint threshold for 'score' must be a finite scalar",
    ):
        constraint_mask(table, {"score": ("<=", threshold)})
    with pytest.raises(
        ValueError,
        match="Constraint threshold for 'score' must be a finite scalar",
    ):
        select_under_constraints(
            table,
            constraints={"score": ("<=", threshold)},
            objective="score",
            direction="min",
        )
    with pytest.raises(
        ValueError,
        match="Constraint threshold for 'score' must be a finite scalar",
    ):
        equal_quality_selection(
            table,
            quality_constraints={"score": ("<=", threshold)},
            compression_objective="score",
        )


@pytest.mark.parametrize(
    ("metric", "direction", "reference"),
    [
        (False, "min", 1.0),
        (np.bool_(False), "min", 1.0),
        ("0.0", "min", 1.0),
        (b"0.0", "min", 1.0),
        (np.str_("0.0"), "min", 1.0),
        (np.array("0.0"), "min", 1.0),
        (True, "max", 0.0),
        (np.bool_(True), "max", 0.0),
        ("1.0", "max", 0.0),
        (b"1.0", "max", 0.0),
        (np.str_("1.0"), "max", 0.0),
        (np.array("1.0"), "max", 0.0),
    ],
)
def test_record_dominates_treats_bool_and_numeric_text_objectives_as_missing(
    metric: object, direction: str, reference: float
) -> None:
    left = {"objective": metric}
    right = {"objective": reference}

    assert not record_dominates(
        left,
        right,
        ["objective"],
        directions={"objective": direction},
    )
    assert not record_dominates(
        left,
        right,
        ["objective"],
        directions={"objective": direction},
        allow_missing=False,
    )


def test_pareto_front_treats_numeric_text_objectives_as_missing() -> None:
    table = pd.DataFrame(
        [
            {"name": "text_fast", "runtime": "0.0"},
            {"name": "numeric_slow", "runtime": 1.0},
        ]
    )

    indices = pareto_front_indices(table, ["runtime"], directions={"runtime": "min"})
    mask = is_pareto_front(table, ["runtime"], directions={"runtime": "min"})

    assert table.loc[indices, "name"].tolist() == ["text_fast", "numeric_slow"]
    assert mask.tolist() == [True, True]
