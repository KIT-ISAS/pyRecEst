import copy
import warnings

import numpy as np

from .abstract_distribution_type import AbstractDistributionType


class AbstractDiracDistribution(AbstractDistributionType):
    def __init__(self, d, w=None):
        if w is None:
            w = np.ones(d.shape[0]) / d.shape[0]

        assert d.shape[0] == w.shape[0], "Number of Diracs and weights must match."
        self.d = copy.copy(d)
        self.w = copy.copy(w)
        self.normalize_in_place()

    def normalize_in_place(self):
        if not np.isclose(np.sum(self.w), 1, atol=1e-10):
            warnings.warn("Weights are not normalized.", RuntimeWarning)
            self.w = self.w / np.sum(self.w)

    def normalize(self):
        dist = copy.deepcopy(self)
        dist.normalize_in_place()
        return dist

    def apply_function(self, f, f_supports_multiple=True):
        dist = copy.deepcopy(self)
        if f_supports_multiple:
            dist.d = f(dist.d)
        else:
            dist.d = np.apply_along_axis(f, 1, dist.d)
        return dist

    def reweigh(self, f):
        w_likelihood = f(self.d)
        assert (
            w_likelihood.shape == self.w.shape
        ), "Function returned wrong number of outputs."
        assert np.all(w_likelihood >= 0)
        assert np.sum(w_likelihood) > 0

        w_posterior_unnormalized = w_likelihood * self.w

        w_posterior_normalized = w_posterior_unnormalized / np.sum(
            w_posterior_unnormalized
        )
        dist = self.__class__(self.d, w_posterior_normalized)
        return dist

    def sample(self, n):
        ids = np.random.choice(self.w.size, size=n, p=self.w)
        return self.d[ids] if self.d.ndim == 1 else self.d[ids, :]

    def entropy(self):
        warnings.warn("Entropy is not defined in a continuous sense")
        return -np.sum(self.w * np.log(self.w))

    def integrate(self, left=None, right=None):
        assert (
            left is None and right is None
        ), "Must overwrite in child class to use integral limits"
        return np.sum(self.w)

    def log_likelihood(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def pdf(self, xs):
        raise NotImplementedError("PDF:UNDEFINED, pdf is not defined")

    def integrate_numerically(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def trigonometric_moment_numerical(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def sample_metropolis_hastings(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def squared_distance_numerical(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def kld_numerical(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def mode(self, rel_tol=0.001):
        highest_val, ind = np.max(self.w), np.argmax(self.w)
        if (highest_val / self.w.size) < (1 + rel_tol):
            warnings.warn(
                "The samples may be equally weighted, .mode is likely to return a bad result."
            )
        return self.d[:, ind]

    def mode_numerical(self, _=None):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def entropy_numerical(self):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    @classmethod
    def from_distribution(cls, distribution, n_samples):
        is_valid_class = False
        # Look if distribution is of the correct type
        for base in cls.__bases__:
            if isinstance(distribution, base):
                is_valid_class = True

        assert is_valid_class
        assert isinstance(n_samples, int) and n_samples > 0
        samples = distribution.sample(n_samples)
        weights = np.ones(n_samples) / n_samples
        return cls(samples, weights)
