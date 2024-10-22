import matplotlib.pyplot as plt

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arctan2, linalg, reshape, sum

from ..circle.circular_dirac_distribution import CircularDiracDistribution
from .abstract_hypersphere_subset_dirac_distribution import (
    AbstractHypersphereSubsetDiracDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


class HypersphericalDiracDistribution(
    AbstractHypersphereSubsetDiracDistribution, AbstractHypersphericalDistribution
):
    def plot(self, *args, **kwargs):
        if self.dim == 2:
            p = plt.stem(arctan2(self.d[1, :], self.d[0, :]), self.w, *args, **kwargs)
        elif self.dim == 3:
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
        assert (
            self.dim == 2
        ), "Conversion to circular dirac distribution only supported for 2D case."
        return CircularDiracDistribution(arctan2(self.d[1, :], self.d[0, :]), self.w)

    def mean_direction(self):
        vec_sum = sum(self.d * reshape(self.w, (-1, 1)), axis=0)
        mu = vec_sum / linalg.norm(vec_sum)
        return mu
