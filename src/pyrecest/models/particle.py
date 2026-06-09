"""Backward-compatible particle-model aliases.

The canonical implementations live in :mod:`pyrecest.models.likelihood`.
This module keeps the historical import path ``pyrecest.models.particle`` from
drifting away from the public model API.
"""

from .likelihood import LikelihoodMeasurementModel, SampleableTransitionModel

__all__ = [
    "LikelihoodMeasurementModel",
    "SampleableTransitionModel",
]
