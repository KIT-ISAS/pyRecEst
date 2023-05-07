import copy
import warnings
from abc import abstractmethod

import numpy as np

from .abstract_distribution import AbstractDistribution
from .hypersphere_subset.abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class CustomDistribution(AbstractDistribution):
    def __init__(self, f, dim: int):
        AbstractDistribution.__init__(self, dim=dim)
        self.f = f
        self.scale_by = 1
        if isinstance(self, AbstractHypersphereSubsetDistribution):
            self.shift_by = None
        else:
            self.shift_by = np.zeros(dim)

    def pdf(self, xs):
        xs = np.asarray(xs)
        p = self.scale_by * self.f(xs - np.atleast_2d(self.shift_by))
        p = np.reshape(p, xs.size // self.dim)
        return p

    def shift(self, shift_vector):
        assert self.dim == np.size(shift_vector) and shift_vector.ndim <= 1
        cd = copy.deepcopy(self)
        cd.shift_by = self.shift_by + shift_vector
        return cd

    @abstractmethod
    def integrate(self, integration_boundaries=None):
        pass

    def normalize(self, verify=None):
        cd = copy.deepcopy(self)

        integral, err = self.integrate()
        cd.scale_by = cd.scale_by / integral

        if verify is None:
            verify = "maxFunEvalsPass" in err

        if verify and abs(cd.integrate()[0] - 1) > 0.001:
            warnings.warn("Density is not yet properly normalized.", UserWarning)

        return cd
