import numpy as np
import matplotlib.pyplot as plt
from ..circle.circular_dirac_distribution import CircularDiracDistribution
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .abstract_hypersphere_subset_dirac_distribution import AbstractHypersphereSubsetDiracDistribution

class HypersphericalDiracDistribution(AbstractHypersphereSubsetDiracDistribution, AbstractHypersphericalDistribution):
    def plot(self, *args, **kwargs):
        if self.dim == 2:
            p = plt.stem(np.atan2(self.d[1, :], self.d[0, :]), self.w, *args, **kwargs)
        elif self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.d[:, 0], self.d[:, 1], self.d[:, 2], c='b', marker='o', s=self.w) 
        else:
            raise NotImplementedError("Plotting for this dimension is currently not supported")
        return p

    def to_circular_dirac_distribution(self):
        assert self.dim == 2, "Conversion to circular dirac distribution only supported for 2D case."
        return CircularDiracDistribution(np.atan2(self.d[1, :], self.d[0, :]), self.w)

    def mean_direction(self):
        vec_sum = np.sum(self.d * np.reshape(self.w, (-1, 1)), axis=0)
        mu = vec_sum / np.linalg.norm(vec_sum)
        return mu
