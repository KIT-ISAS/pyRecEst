from pyrecest.backend import reshape
from pyrecest.backend import prod
from pyrecest.backend import array
from pyrecest.backend import diff
from scipy.integrate import nquad

from ..abstract_bounded_nonperiodic_distribution import (
    AbstractBoundedNonPeriodicDistribution,
)


class AbstractHyperrectangularDistribution(AbstractBoundedNonPeriodicDistribution):
    def __init__(self, bounds):
        AbstractBoundedNonPeriodicDistribution.__init__(self, np.size(bounds[0]))
        self.bounds = bounds

    def get_manifold_size(self):
        s = prod(diff(self.bounds, axis=1))
        return s

    @property
    def input_dim(self):
        return self.dim

    def integrate(self, integration_boundaries=None) -> float:
        """
        Integrate the probability density function over given boundaries.
        If no boundaries are provided, default to `self.bounds`.

        Args:
            integration_boundaries (tuple): A tuple of two elements, each of which can be either a scalar or an array.
                If a scalar, it represents a single boundary value.
                If an array, it represents multiple boundary values.

        Returns:
            float: The result of the integration.
        """
        if integration_boundaries is None:
            integration_boundaries = self.bounds

        integration_boundaries = reshape(integration_boundaries, (2, -1))
        left, right = integration_boundaries

        integration_boundaries = zip(left, right)
        return nquad(lambda *args: self.pdf(array(args)), integration_boundaries)[0]