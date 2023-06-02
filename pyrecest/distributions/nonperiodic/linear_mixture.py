import warnings
from typing import List
from beartype import beartype

from ..abstract_mixture import AbstractMixture
from .abstract_linear_distribution import AbstractLinearDistribution
from .gaussian_distribution import GaussianDistribution
import numpy as np

class LinearMixture(AbstractMixture, AbstractLinearDistribution):
    @beartype
    def __init__(self, dists: List[AbstractLinearDistribution], w: np.ndarray):
        assert all(isinstance(dist, AbstractLinearDistribution) for dist in dists), "dists must be a list of linear distributions"
        if all(isinstance(dist, GaussianDistribution) for dist in dists):
            warnings.warn("For mixtures of Gaussians, consider using GaussianMixture.", UserWarning)
        AbstractLinearDistribution.__init__(self, dists[0].dim)
        AbstractMixture.__init__(self, dists, w)

    @property
    def input_dim(self):
        return AbstractLinearDistribution.input_dim.fget(self)
