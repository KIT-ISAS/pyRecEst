import warnings
from abc import abstractmethod

import numpy as np

from .abstract_distribution_type import AbstractDistributionType


class AbstractMixture(AbstractDistributionType):
    def __init__(self, dists, w=None):
        AbstractDistributionType.__init__(self)
        if w is None:
            w = np.ones(len(dists)) / len(dists)
        else:
            w = np.asarray(w)

        assert len(dists) == len(w), "Sizes of dists and w must be equal"
        assert all(
            dists[0].dim == dist.dim for dist in dists
        ), "All distributions must have the same dimension"

        non_zero_indices = np.nonzero(w)[0]

        if len(non_zero_indices) < len(w):
            warnings.warn(
                "Elements with zero weights detected. Pruning elements in mixture with weight zero."
            )
            dists = [dists[i] for i in non_zero_indices]
            w = w[non_zero_indices]

        self.dists = dists

        if abs(np.sum(w) - 1) > 1e-10:
            warnings.warn("Weights of mixture do not sum to one.")
            self.w = w / np.sum(w)
        else:
            self.w = w

    def sample(self, n):
        d = np.random.choice(len(self.w), size=n, p=self.w)

        occurrences = np.bincount(d, minlength=len(self.dists))
        count = 0
        s = np.empty((n, self.input_dim))
        for i, occ in enumerate(occurrences):
            if occ != 0:
                s[count : count + occ, :] = self.dists[i].sample(occ)  # noqa: E203
                count += occ

        order = np.argsort(d)
        s = s[order, :]  # noqa: E203

        return s

    @property
    @abstractmethod
    def input_dim(self):
        pass

    def pdf(self, xs):
        assert xs.shape[-1] == self.input_dim, "Dimension mismatch"

        p = np.zeros(1) if xs.ndim == 1 else np.zeros(xs.shape[0])

        for i, dist in enumerate(self.dists):
            p += self.w[i] * dist.pdf(xs)

        return p
