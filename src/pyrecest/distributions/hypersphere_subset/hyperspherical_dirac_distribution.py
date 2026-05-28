import matplotlib.pyplot as plt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arctan2, reshape, sum

from ..circle.circular_dirac_distribution import CircularDiracDistribution
from .abstract_hypersphere_subset_dirac_distribution import (
    AbstractHypersphereSubsetDiracDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class HypersphericalDiracDistribution(
    AbstractHypersphereSubsetDiracDistribution, AbstractHypersphericalDistribution
):
    def plot(self, *args, **kwargs):
        if self.dim == 1:
            p = plt.stem(arctan2(self.d[:, 1], self.d[:, 0]), self.w, *args, **kwargs)
        elif self.dim == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            p = ax.scatter(
                self.d[:, 0], self.d[:, 1], self.d[:, 2], c="b", marker="o", s=self.w
            )
        else:
            raise NotImplementedError(
                "Plotting for this dimension is currently not supported"
            )
        return p

    def to_circular_dirac_distribution(self):
        if self.dim != 1:
            raise ValueError(
                "Conversion to circular Dirac distribution is only supported for S¹."
            )
        return CircularDiracDistribution(arctan2(self.d[:, 1], self.d[:, 0]), self.w)

    def mean_direction(self):
        mean_res_vec = self.mean_resultant_vector()
        return self._normalize_mean_direction(mean_res_vec)

    def mean_resultant_vector(self):
        return sum(self.d * reshape(self.w, (-1, 1)), axis=0)
