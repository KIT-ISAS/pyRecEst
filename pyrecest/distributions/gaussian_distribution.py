import copy

import numpy as np
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal as mvn

from .abstract_linear_distribution import AbstractLinearDistribution


class GaussianDistribution(AbstractLinearDistribution):
    def __init__(self, mu, C, check_validity=True):
        assert mu.shape[0] == C.shape[0] == C.shape[1], "Size of C invalid"
        self.dim = mu.shape[0]
        assert mu.ndim <= 1
        self.mu = mu

        if check_validity:
            if self.dim == 1:
                assert C > 0, "C must be positive definite"
            elif self.dim == 2:
                assert (
                    C[0, 0] > 0 and np.linalg.det(C) > 0
                ), "C must be positive definite"
            else:
                cholesky(C)  # Will fail if C is not positive definite

        self.C = C

    def pdf(self, xa):
        assert (
            xa.shape[-1] == self.mu.shape[0] or self.mu.size == 1
        ), "Dimension incorrect"
        return mvn.pdf(xa, self.mu.T, self.C).T

    def shift(self, offsets):
        assert offsets.size == self.dim
        new_gaussian = copy.deepcopy(self)
        new_gaussian.mu = self.mu + offsets
        return new_gaussian

    def mean(self):
        return self.mu

    def mode(self):
        return self.mu

    def set_mode(self, new_mode):
        new_dist = copy.deepcopy(self)
        new_dist.mu = new_mode
        return new_dist

    def covariance(self):
        return self.C

    def multiply(self, other):
        assert self.dim == other.dim
        K = np.linalg.solve(self.C + other.C, self.C)
        new_mu = self.mu + np.dot(K, (other.mu - self.mu))
        new_C = self.C - np.dot(K, self.C)
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def convolve(self, other):
        assert self.dim == other.dim
        new_mu = self.mu + other.mu
        new_C = self.C + other.C
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def marginalize_out(self, dimensions):
        if type(dimensions) is int:  # Make it iterable if single integer
            dimensions = [dimensions]
        assert all(dim <= self.dim for dim in dimensions)
        remaining_dims = [i for i in range(self.dim) if i not in dimensions]
        new_mu = self.mu[remaining_dims]
        new_C = self.C[np.ix_(remaining_dims, remaining_dims)]
        return GaussianDistribution(new_mu, new_C, check_validity=False)

    def sample(self, n):
        return np.random.multivariate_normal(self.mu, self.C, n)
