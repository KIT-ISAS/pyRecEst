from __future__ import annotations

import math

import numpy as np
import pytest
from pyrecest.cli import _check_expected_mapping, _validate_tolerance


def test_validate_tolerance_accepts_finite_nonnegative_numbers() -> None:
    assert _validate_tolerance(0.0) == 0.0
    assert _validate_tolerance(1e-8) == 1e-8
    assert _validate_tolerance(np.float64(1e-8)) == 1e-8
    assert _validate_tolerance(np.array(1e-8)) == 1e-8


def test_validate_tolerance_rejects_invalid_values() -> None:
    invalid_tolerances = (
        math.nan,
        math.inf,
        -1.0,
        True,
        "0.1",
        np.bool_(True),
        np.array(True, dtype=object),
        np.array("0.1", dtype=object),
        np.array([0.1]),
    )
    for tolerance in invalid_tolerances:
        with pytest.raises(ValueError, match="tolerance"):
            _validate_tolerance(tolerance)


def test_expected_mapping_rejects_nan_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        _check_expected_mapping(
            "metrics",
            {"rmse": 1.0},
            {"rmse": 2.0},
            tolerance=math.nan,
        )


def test_expected_mapping_accepts_finite_numeric_actuals() -> None:
    assert (
        _check_expected_mapping(
            "metrics",
            {"rmse": 1.000001},
            {"rmse": 1.0},
            tolerance=1e-5,
        )
        == []
    )


@pytest.mark.parametrize("actual_value", [True, "1.0", math.nan, math.inf])
def test_expected_mapping_rejects_malformed_numeric_actuals(actual_value) -> None:
    errors = _check_expected_mapping(
        "metrics",
        {"rmse": actual_value},
        {"rmse": 1.0},
        tolerance=0.0,
    )

    assert errors
    assert "metrics.rmse mismatch" in errors[0]
