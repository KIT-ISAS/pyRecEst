import numpy as np
import matplotlib.pyplot as plt
from .abstract_linear_distribution import AbstractLinearDistribution
from ..abstract_dirac_distribution import AbstractDiracDistribution

class LinearDiracDistribution(AbstractDiracDistribution, AbstractLinearDistribution): # type: ignore[misc]

    def mean(self):
        return np.average(self.d, weights=self.w, axis=0)

    def covariance(self):
        _, C = LinearDiracDistribution.weighted_samples_to_mean_and_cov(self.d, self.w)
        return C

    def plot(self, *args, **kwargs):
        if self.dim == 1:
            plt.stem(self.d, self.w, *args, **kwargs)
        elif self.dim == 2:
            plt.scatter(self.d[0, :], self.d[1, :], self.w / max(self.w) * 100, *args, **kwargs)
        elif self.dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.d[0, :], self.d[1, :], self.w / max(self.w) * 100, *args, **kwargs)
        else:
            raise ValueError("Plotting not supported for this dimension")

    @staticmethod
    def from_distribution(distribution, no_of_samples):
        samples = distribution.sample(no_of_samples)
        return LinearDiracDistribution(samples, np.ones(no_of_samples) / no_of_samples)

    @staticmethod
    def weighted_samples_to_mean_and_cov(samples, weights=None):
        if weights is None:
            weights = np.ones(samples.shape[1]) / samples.shape[1]

        mean = np.average(samples, weights=weights, axis=0)
        deviation = samples - mean
        covariance = np.cov(deviation.T, aweights=weights, bias=True)

        return mean, covariance
