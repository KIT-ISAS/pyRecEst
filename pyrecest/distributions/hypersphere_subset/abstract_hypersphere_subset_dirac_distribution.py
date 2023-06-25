import numpy as np

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
        C = self.d.T * self.w @ self.d
        return C

    def entropy(self):
        result = -np.sum(self.w * np.log(self.w))
        return result

    def integrate(self):
        pass
