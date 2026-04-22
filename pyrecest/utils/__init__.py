"""Utility helpers for :mod:`pyrecest`."""

import numpy as _np

from . import multisession_assignment as _multisession_assignment
from .multisession_assignment import (
    MultiSessionAssignmentResult,
    solve_multisession_assignment,
    tracks_to_session_labels,
)

from .nonrigid_point_set_registration import (
    ThinPlateSplineRegistrationResult,
    ThinPlateSplineTransform,
    estimate_thin_plate_spline,
    joint_tps_registration_assignment,
)

_multisession_assignment.np = _np

__all__ = [
    "MultiSessionAssignmentResult",
    "solve_multisession_assignment",
    "tracks_to_session_labels",
    "ThinPlateSplineRegistrationResult",
    "ThinPlateSplineTransform",
    "estimate_thin_plate_spline",
    "joint_tps_registration_assignment",
]
