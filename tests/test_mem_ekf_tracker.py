from pyrecest.backend import allclose, array, to_numpy
from pyrecest.filters.mem_ekf_tracker import MEMEKFTracker


def _as_bool(value):
    converted = to_numpy(value)
    if hasattr(converted, "item"):
        return bool(converted.item())
    return bool(converted)


def test_mem_ekf_pseudo_measurement_covariance_matches_gaussian_moments():
    covariance = array([[2.0, 0.5], [0.5, 3.0]])
    expected = array(
        [
            [8.0, 2.0, 0.5],
            [2.0, 6.25, 3.0],
            [0.5, 3.0, 18.0],
        ]
    )

    result = MEMEKFTracker._pseudo_measurement_covariance(covariance)

    assert _as_bool(allclose(result, expected))


def test_mem_ekf_pseudo_measurement_covariance_is_centered_not_raw_moment():
    covariance = array([[2.0, 0.0], [0.0, 3.0]])
    expected = array(
        [
            [8.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 18.0],
        ]
    )

    result = MEMEKFTracker._pseudo_measurement_covariance(covariance)

    assert _as_bool(allclose(result, expected))
