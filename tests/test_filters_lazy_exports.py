from __future__ import annotations

import importlib
import sys


def test_filters_all_is_unique() -> None:
    import pyrecest.filters as filters

    assert len(filters.__all__) == len(set(filters.__all__))


def test_filters_namespace_resolves_baseline_filter_lazily() -> None:
    sys.modules.pop("pyrecest.filters", None)
    sys.modules.pop("pyrecest.filters.kalman_filter", None)
    sys.modules.pop("pyrecest.filters.spherical_harmonics_eot_tracker", None)

    filters = importlib.import_module("pyrecest.filters")

    assert "KalmanFilter" in filters.__all__
    assert "pyrecest.filters.kalman_filter" not in sys.modules
    assert "pyrecest.filters.spherical_harmonics_eot_tracker" not in sys.modules

    assert filters.KalmanFilter.__name__ == "KalmanFilter"
    assert "pyrecest.filters.kalman_filter" in sys.modules
    assert "pyrecest.filters.spherical_harmonics_eot_tracker" not in sys.modules
