from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import TrackingEvent, event_from_measurement, record_from_update

_NUMERIC_BYTES = bytes.fromhex("312e30")


@pytest.mark.parametrize(
    "measurement",
    ([True], ["1.0"], [_NUMERIC_BYTES], np.array([1.0 + 2.0j])),
)
def test_tracking_event_rejects_non_real_numeric_measurements(measurement) -> None:
    with pytest.raises(
        ValueError, match="measurement must contain real-valued numeric entries"
    ):
        TrackingEvent(time=0.0, source="rf", measurement=measurement)


@pytest.mark.parametrize(
    "covariance",
    (
        [[True]],
        [["1.0"]],
        [[_NUMERIC_BYTES]],
        np.array([[1.0 + 2.0j]]),
    ),
)
def test_tracking_event_rejects_non_real_numeric_covariances(covariance) -> None:
    with pytest.raises(
        ValueError, match="covariance must contain real-valued numeric entries"
    ):
        TrackingEvent(time=0.0, source="rf", measurement=[0.0], covariance=covariance)


def test_tracking_record_rejects_non_real_numeric_state_fields() -> None:
    event = event_from_measurement(time=0.0, source="rf")
    base_kwargs = {
        "event": event,
        "prior_mean": [0.0, 0.0],
        "prior_cov": np.eye(2),
        "posterior_mean": [0.0, 0.0],
        "posterior_cov": np.eye(2),
    }

    invalid_cases = (
        ("prior_mean", [True, 0.0], "prior_mean"),
        ("prior_cov", [["1.0", "0.0"], ["0.0", "1.0"]], "prior_cov"),
        ("posterior_mean", np.array([1.0 + 2.0j, 0.0 + 0.0j]), "posterior_mean"),
        (
            "posterior_cov",
            np.array([[1.0 + 2.0j, 0.0], [0.0, 1.0]]),
            "posterior_cov",
        ),
        ("innovation", [_NUMERIC_BYTES], "innovation"),
        ("innovation_cov", [[_NUMERIC_BYTES]], "innovation_cov"),
    )

    for field, value, message_name in invalid_cases:
        kwargs = dict(base_kwargs)
        kwargs[field] = value
        with pytest.raises(
            ValueError,
            match=f"{message_name} must contain real-valued numeric entries",
        ):
            record_from_update(**kwargs)
