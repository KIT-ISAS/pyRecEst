from __future__ import annotations

import numpy as np
from pyrecest.tracking import innovation_diagnostic


def test_innovation_diagnostic_parses_explicit_status_string() -> None:
    explicit_status = innovation_diagnostic(
        np.array([10.0]),
        np.eye(1),
        gate_threshold=9.0,
        accepted="0",
    )

    assert explicit_status.accepted is False
