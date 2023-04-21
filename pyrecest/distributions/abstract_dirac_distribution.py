import copy
import warnings

import numpy as np

from .abstract_distribution import AbstractDistribution


class AbstractDiracDistribution(AbstractDistribution):
    def __init__(self, d, w=None):
        self.dim = d.shape[-1]
        if self.dim > d.shape[0]:
            warnings.warn(
                "Not even one Dirac per dimension. If this warning is unexpected, verify d_ is shaped correctly."
            )
        if w is None:
            w = np.ones(d.shape[0]) / d.shape[0]

        assert d.shape[0] == w.shape[0], "Number of Diracs and weights must match."
        self.d = copy.copy(d)
        self.w = copy.copy(w)
        self._normalize_in_place()

    def _normalize_in_place(self):
        if not np.isclose(np.sum(self.w), 1, atol=1e-10):
            warnings.warn("Weights are not normalized.", RuntimeWarning)
            self.w = self.w / np.sum(self.w)

    def normalize(self):
        dist = copy.deepcopy(self)
        dist._normalize_in_place()
        return dist

    def apply_function(self, f):
        d_ = np.zeros_like(self.d)
        for i in range(self.d.shape[0]):
            d_[i, :] = f(self.d[i, :])

        dist = copy.deepcopy(self)
        dist.d = d_
        dist.w = self.w
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
        return self.d[ids, :]

    def entropy(self):
        print("Entropy is not defined in a continuous sense")
        return -np.sum(self.w * np.log(self.w))

    def integrate(self, left=None, right=None):
        assert (
            left is None and right is None
        ), "Must overwrite in child class to use integral limits"
        return np.sum(self.w)

    def log_likelihood(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def pdf(self, *args):
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
            print(
                "The samples may be equally weighted, .mode is likely to return a bad result."
            )
        return self.d[:, ind]

    def mode_numerical(self):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def entropy_numerical(self):
        raise NotImplementedError("PDF:UNDEFINED, not supported")
