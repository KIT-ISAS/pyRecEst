import pytest
from pyrecest.calibration.time_offset import aggregate_time_offset_sweeps


def test_aggregate_sweeps_rejects_ragged_offset_values_with_stable_error():
    with pytest.raises(ValueError, match="time_offset_s must be a real scalar"):
        aggregate_time_offset_sweeps(
            [[{"time_offset_s": [[0.0], [1.0, 2.0]], "count": 1.0, "rmse": 0.0}]]
        )
