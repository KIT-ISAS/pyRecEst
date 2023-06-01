from beartype import beartype
import numpy as np

from .abstract_custom_nonperiodic_distribution import (
    AbstractCustomNonPeriodicDistribution,
)
from .nonperiodic.abstract_hyperrectangular_distribution import (
    AbstractHyperrectangularDistribution,
)
from typing import Callable

class CustomHyperrectangularDistribution(
    AbstractHyperrectangularDistribution, AbstractCustomNonPeriodicDistribution
):
    @beartype
    def __init__(self, f: Callable, bounds: np.ndarray):
        AbstractHyperrectangularDistribution.__init__(self, bounds)
        AbstractCustomNonPeriodicDistribution.__init__(self, f)

    @beartype
    def pdf(self, xs: np.ndarray) -> np.ndarray:
        return AbstractCustomNonPeriodicDistribution.pdf(self, xs)
