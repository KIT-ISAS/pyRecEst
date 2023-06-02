import numpy as np
from typing import List, Union

from .linear_mixture import LinearMixture
from .linear_dirac_distribution import LinearDiracDistribution
from .gaussian_distribution import GaussianDistribution
from beartype import beartype
import numbers

class GaussianMixture(LinearMixture):
    @beartype
    def __init__(self, dists: List[GaussianDistribution], w: np.ndarray):
        LinearMixture.__init__(self, dists, w)

    def mean(self):
        gauss_array = self.dists
        return np.dot(np.array([g.mu for g in gauss_array]), self.w)
            
    @beartype
    def set_mean(self, new_mean: Union[np.ndarray, numbers.Real]):
        mean_offset = new_mean - self.mean()
        for dist in self.dists:
            dist.mu += mean_offset

    def to_gaussian(self):
        gauss_array = self.dists
        mu, C = self.mixture_parameters_to_gaussian_parameters(
            np.array([g.mu for g in gauss_array]),
            np.stack([g.C for g in gauss_array], axis=2),
            self.w,
        )
        return GaussianDistribution(mu, C)

    def covariance(self):
        gauss_array = self.dists
        _, C = self.mixture_parameters_to_gaussian_parameters(
            np.array([g.mu for g in gauss_array]),
            np.stack([g.C for g in gauss_array], axis=2),
            self.w,
        )
        return C

    @staticmethod
    def mixture_parameters_to_gaussian_parameters(
        means, covariance_matrices, weights=None
    ):
        if weights is None:
            weights = np.ones(means.shape[1]) / means.shape[1]

        C_from_cov = np.sum(covariance_matrices * weights.reshape(1, 1, -1), axis=2)
        mu, C_from_means = LinearDiracDistribution.weighted_samples_to_mean_and_cov(
            means, weights
        )
        C = C_from_cov + C_from_means

        return mu, C
