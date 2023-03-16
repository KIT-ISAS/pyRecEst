import numpy as np
import warnings
import copy

class AbstractDiracDistribution:
    def __init__(self, d_, w_=None):
        self.dim = d_.shape[0]
        if self.dim > d_.shape[1]:
            print("Not even one Dirac per dimension. If this warning is unexpected, verify d_ is shaped correctly.")
        if w_ is None:
            w_ = np.ones(d_.shape[1]) / d_.shape[1]
        else:
            w_ = np.asarray(w_).reshape(-1)  # Ensure w_ has shape (n,)
        assert d_.shape[1] == w_.shape[0], "Number of Diracs and weights must match."
        self.d = d_
        self.w = w_
        self = self.normalize()

    def normalize(self):
        dist = copy.deepcopy(self)
        if not np.isclose(np.sum(self.w), 1, atol=1e-10):
            warnings.warn("Weights are not normalized.", RuntimeWarning)
            dist.w = self.w / np.sum(self.w)
        return dist

    def apply_function(self, f):
        d_ = np.zeros_like(self.d)
        for i in range(self.d.shape[1]):
            d_[:, i] = f(self.d[:, i])
        dist = self.__class__(d_, self.w)
        return dist

    def reweigh(self, f):
        w_new = f(self.d)
        assert w_new.shape == self.w.shape, "Function returned wrong number of outputs."
        assert np.all(w_new >= 0)
        assert np.sum(w_new) > 0

        dist = self.__class__(self.d, w_new * self.w)
        dist.w /= np.sum(dist.w)
        return dist

    def sample(self, n):
        ids = np.random.choice(self.w.size, size=n, p=self.w)
        return self.d[:, ids]

    def entropy(self):
        print("Entropy is not defined in a continuous sense")
        return -np.sum(self.w * np.log(self.w))

    def integral(self):
        return np.sum(self.w)
    
    def log_likelihood(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def pdf(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, pdf is not defined")

    def integral_numerical(self, *args):
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
            print("The samples may be equally weighted, .mode is likely to return a bad result.")
        return self.d[:, ind]

    def mode_numerical(self):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def entropy_numerical(self):
        raise NotImplementedError("PDF:UNDEFINED, not supported")
