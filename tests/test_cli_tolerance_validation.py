from __future__ import annotations

import math

import pytest
from pyrecest.cli import _check_expected_mapping, _validate_tolerance


def test_validate_tolerance_accepts_finite_nonnegative_numbers() -> None:
    assert _validate_tolerance(0.0) == 0.0
    assert _validate_tolerance(1e-8) == 1e-8


def test_validate_tolerance_rejects_invalid_values() -> None:
    for tolerance in (math.nan, math.inf, -1.0, True, "0.1"):
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


def test_expected_mapping_compares_booleans_exactly() -> None:
    assert (
        _check_expected_mapping(
            "diagnostics",
            {"flag": True},
            {"flag": True},
            tolerance=0.0,
        )
        == []
    )

    failures = _check_expected_mapping(
        "diagnostics",
        {"flag": 1},
        {"flag": True},
        tolerance=0.0,
    )

    assert failures == ["diagnostics.flag mismatch: expected True, got 1"]


def test_expected_mapping_rejects_boolean_actual_for_numeric_expected() -> None:
    failures = _check_expected_mapping(
        "metrics",
        {"count": True},
        {"count": 1},
        tolerance=0.0,
    )

    assert failures == ["metrics.count mismatch: expected numeric 1, got True"]
