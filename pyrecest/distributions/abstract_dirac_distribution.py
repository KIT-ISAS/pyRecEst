import copy
import warnings
from collections.abc import Callable

import numpy as np
from beartype import beartype

from .abstract_distribution_type import AbstractDistributionType


class AbstractDiracDistribution(AbstractDistributionType):
    """
    This class represents an abstract base for Dirac distributions.
    """

    @beartype
    def __init__(self, d: np.ndarray, w: np.ndarray | float | np.float64 | None = None):
        """
        Initialize a Dirac distribution with given Dirac locations and weights.

        :param d: Dirac locations as a numpy array.
        :param w: Weights of Dirac locations as a numpy array. If not provided, defaults to uniform weights.
        """
        if w is None:
            w = np.ones(d.shape[0]) / d.shape[0]

        assert d.shape[0] == np.size(w), "Number of Diracs and weights must match."
        self.d = copy.copy(d)
        self.w = copy.copy(w)
        self.normalize_in_place()

    @beartype
    def normalize_in_place(self):
        """
        Normalize the weights in-place to ensure they sum to 1.
        """
        if not np.isclose(np.sum(self.w), 1, atol=1e-10):
            warnings.warn("Weights are not normalized.", RuntimeWarning)
            self.w = self.w / np.sum(self.w)

    @beartype
    def normalize(self) -> "AbstractDiracDistribution":
        dist = copy.deepcopy(self)
        dist.normalize_in_place()
        return dist

    @beartype
    def apply_function(
        self, f: Callable, f_supports_multiple: bool = True
    ) -> "AbstractDiracDistribution":
        """
        Apply a function to the Dirac locations and return a new distribution.

        :param f: Function to apply.
        :returns: A new distribution with the function applied to the locations.
        """
        dist = copy.deepcopy(self)
        if f_supports_multiple:
            dist.d = f(dist.d)
        else:
            dist.d = np.apply_along_axis(f, 1, dist.d)
        return dist

    @beartype
    def reweigh(self, f: Callable) -> "AbstractDiracDistribution":
        wNew = f(self.d)

        assert wNew.shape == (
            self.d.shape[0],
        ), "Function returned wrong number of outputs."
        assert np.all(wNew >= 0), "All weights should be greater than or equal to 0."
        assert np.sum(wNew) > 0, "The sum of all weights should be greater than 0."

        self.w = wNew * self.w
        self.w = self.w / np.sum(self.w)

        return self

    @beartype
    def sample(self, n: int | np.int32 | np.int64) -> np.ndarray:
        ids = np.random.choice(np.size(self.w), size=n, p=self.w)
        return self.d[ids] if self.d.ndim == 1 else self.d[ids, :]  # noqa: E203

    def entropy(self) -> float:
        warnings.warn("Entropy is not defined in a continuous sense")
        return -np.sum(self.w * np.log(self.w))

    def integrate(self, left=None, right=None) -> float:
        assert (
            left is None and right is None
        ), "Must overwrite in child class to use integral limits"
        return np.sum(self.w)

    def log_likelihood(self, *args):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def pdf(self, _):
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
        return self.d[ind, :]

    def mode_numerical(self, _=None):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    def entropy_numerical(self):
        raise NotImplementedError("PDF:UNDEFINED, not supported")

    @classmethod
    def is_valid_for_conversion(cls, distribution):
        return any(isinstance(distribution, base) for base in cls.__bases__)

    @classmethod
    def from_distribution(cls, distribution, n_particles):
        assert cls.is_valid_for_conversion(distribution)
        samples = distribution.sample(n_particles)
        return cls(samples)
