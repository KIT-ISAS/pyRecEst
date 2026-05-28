from collections.abc import Callable

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, reshape

from .abstract_custom_nonperiodic_distribution import (
    AbstractCustomNonPeriodicDistribution,
)
from .nonperiodic.abstract_hyperrectangular_distribution import (
    AbstractHyperrectangularDistribution,
)


class CustomHyperrectangularDistribution(
    AbstractHyperrectangularDistribution, AbstractCustomNonPeriodicDistribution
):
    def __init__(self, f: Callable, bounds):
        AbstractHyperrectangularDistribution.__init__(self, bounds)
        AbstractCustomNonPeriodicDistribution.__init__(self, f)

    def pdf(self, xs):
        xs = self._coerce_points(xs)
        return AbstractCustomNonPeriodicDistribution.pdf(self, xs)

    def _coerce_points(self, xs):
        xs = array(xs)
        if xs.ndim == 0:
            if self.dim != 1:
                raise ValueError("Scalar points are only valid for dim == 1")
            return reshape(xs, (1, 1))
        if xs.ndim == 1:
            if self.dim == 1:
                return reshape(xs, (-1, 1))
            if xs.shape[0] == self.dim:
                return reshape(xs, (1, self.dim))
            raise ValueError(
                f"Point dimension {xs.shape[0]} does not match dim {self.dim}"
            )
        if xs.ndim != 2 or xs.shape[1] != self.dim:
            raise ValueError(f"xs must have shape (n, {self.dim})")
        return xs
