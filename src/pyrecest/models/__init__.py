"""Reusable model objects for recursive estimation.

The model package contains filter-independent descriptions of transition and
measurement models. Filters can consume these objects as adapters while their
existing matrix/function APIs remain available.
"""

from .additive_noise import AdditiveNoiseMeasurementModel, AdditiveNoiseTransitionModel

__all__ = [
    "AdditiveNoiseMeasurementModel",
    "AdditiveNoiseTransitionModel",
]
