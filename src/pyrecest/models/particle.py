from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleableTransitionModel:
    """Transition model that propagates particle locations."""

    sample_next: Callable
    function_is_vectorized: bool = True


@dataclass(frozen=True)
class LikelihoodMeasurementModel:
    """Measurement model represented by a likelihood callable."""

    likelihood: Callable
