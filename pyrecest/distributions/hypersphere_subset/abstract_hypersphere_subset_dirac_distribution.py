# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import log, outer, sum, zeros

from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHypersphereSubsetDiracDistribution(
    AbstractDiracDistribution, AbstractHypersphereSubsetDistribution
):
    def __init__(self, d, w=None):
        AbstractHypersphereSubsetDistribution.__init__(self, d.shape[-1] - 1)
        AbstractDiracDistribution.__init__(self, d, w=w)

    def moment(self):
        # Compute the weighted moment matrix
        moment_matrix = zeros(
            (self.d.shape[1], self.d.shape[1])
        )  # Initialize (dim, dim) matrix
        for i in range(self.d.shape[0]):  # Iterate over samples
            moment_matrix += self.w[i] * outer(self.d[i, :], self.d[i, :])

        return moment_matrix

    def entropy(self):
        result = -sum(self.w * log(self.w))
        return result

    def integrate(self, integration_boundaries=None):
        raise NotImplementedError()
