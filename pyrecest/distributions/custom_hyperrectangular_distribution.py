from collections.abc import Callable

import numpy as np
from beartype import beartype

from .abstract_custom_nonperiodic_distribution import (
    AbstractCustomNonPeriodicDistribution,
)
from .nonperiodic.abstract_hyperrectangular_distribution import (
    AbstractHyperrectangularDistribution,
)


class CustomHyperrectangularDistribution(
    AbstractHyperrectangularDistribution, AbstractCustomNonPeriodicDistribution
):
    def __init__(self, f: Callable, bounds: np.ndarray):
        AbstractHyperrectangularDistribution.__init__(self, bounds)
        AbstractCustomNonPeriodicDistribution.__init__(self, f)

    def pdf(self, xs: np.ndarray) -> np.ndarray:
        return AbstractCustomNonPeriodicDistribution.pdf(self, xs)