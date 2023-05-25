import copy
from typing import Callable, Optional

import numpy as np

from ..distributions.nonperiodic.abstract_linear_distribution import (
    AbstractLinearDistribution,
)
from ..distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from .abstract_euclidean_filter import AbstractEuclideanFilter
from .abstract_particle_filter import AbstractParticleFilter


class EuclideanParticleFilter(AbstractParticleFilter, AbstractEuclideanFilter):
    def __init__(self, n_particles: int, dim: int):
        assert (
            isinstance(n_particles, int) and n_particles > 0
        ), "n_particles must be a positive integer"
        assert isinstance(dim, int) and dim > 0, "dim must be a positive integer"
        AbstractParticleFilter.__init__(
            self, LinearDiracDistribution(np.zeros((n_particles, dim)))
        )
        AbstractEuclideanFilter.__init__(
            self, LinearDiracDistribution(np.zeros((n_particles, dim)))
        )

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if not isinstance(new_state, LinearDiracDistribution):
            dist_dirac = LinearDiracDistribution.from_distribution(
                new_state, self._filter_state.d.shape[0]
            )
        else:
            dist_dirac = copy.deepcopy(new_state)

        assert self._filter_state.d.shape == dist_dirac.d.shape

        self._filter_state = dist_dirac

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution: Optional[AbstractLinearDistribution] = None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = False,
    ):
        AbstractParticleFilter.predict_nonlinear(
            self, f, noise_distribution, function_is_vectorized, shift_instead_of_add
        )
