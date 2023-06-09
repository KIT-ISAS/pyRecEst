import warnings

import numpy as np
from beartype import beartype

from ..abstract_mixture import AbstractMixture
from .abstract_linear_distribution import AbstractLinearDistribution
from .gaussian_distribution import GaussianDistribution


class LinearMixture(AbstractMixture, AbstractLinearDistribution):
    @beartype
    def __init__(self, dists: list[AbstractLinearDistribution], w: np.ndarray):
        from .gaussian_mixture import GaussianMixture

        assert all(
            isinstance(dist, AbstractLinearDistribution) for dist in dists
        ), "dists must be a list of linear distributions"
        if not isinstance(self, GaussianMixture) and all(
            isinstance(dist, GaussianDistribution) for dist in dists
        ):
            warnings.warn(
                "For mixtures of Gaussians, consider using GaussianMixture.",
                UserWarning,
            )
        AbstractLinearDistribution.__init__(self, dists[0].dim)
        AbstractMixture.__init__(self, dists, w)

    @property
    def input_dim(self):
        return AbstractLinearDistribution.input_dim.fget(self)
