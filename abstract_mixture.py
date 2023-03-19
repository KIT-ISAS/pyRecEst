import numpy as np
import warnings
from abstract_distribution import AbstractDistribution

class AbstractMixture(AbstractDistribution):
    def __init__(self, dists, w):
        assert len(dists) == len(w), "Size of dists and w must be equal"
        assert all(dists[0].dim == dist.dim for dist in dists), "All distributions must have the same dimension"
        
        self.dim = dists[0].dim
        
        self.dists = [dist for dist, weight in zip(dists, w) if weight != 0]
        self.w = np.array([weight for weight in w if weight != 0])
        
        if abs(sum(self.w) - 1) > 1e-10:
            warnings.warn("Weights of mixture must sum to 1.")
            self.w /= sum(self.w)

    def pdf(self, xa):
        assert xa.shape[0] == self.dim, "Dimension mismatch"
        
        p = np.zeros(xa.shape[1])
        for dist, weight in zip(self.dists, self.w):
            p += weight * dist.pdf(xa)
        
        return p

    def sample(self, n):
        d = np.random.choice(len(self.w), size=n, p=self.w)
        """
        if isinstance(self.dists[0], SE2BinghamDistribution):
            s = np.zeros((self.dim + 1, n))
        else:
            s = np.zeros((self.dim, n))
        """
        
        occurrences = np.bincount(d, minlength=len(self.dists))
        count = 0
        for i, occ in enumerate(occurrences):
            if occ != 0:
                s[:, count:count + occ] = self.dists[i].sample(occ)
                count += occ
        
        order = np.argsort(d)
        s = s[:, order]
        
        return s
