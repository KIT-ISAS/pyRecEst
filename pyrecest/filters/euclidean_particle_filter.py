import copy
from typing import Callable, Optional, Union

import numpy as np

from ..distributions.nonperiodic.abstract_linear_distribution import (
    AbstractLinearDistribution,
)
from ..distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from .abstract_euclidean_filter import AbstractEuclideanFilter
from .abstract_particle_filter import AbstractParticleFilter
from beartype import beartype
from typing import Union

class EuclideanParticleFilter(AbstractParticleFilter, AbstractEuclideanFilter):
    """Euclidean Particle Filter Class."""

    def __init__(self, n_particles: Union[int, np.int64], dim: Union[int, np.int64]):
        if not (isinstance(n_particles, int) and n_particles > 0):
            raise ValueError("n_particles must be a positive integer")
        if not (isinstance(dim, int) and dim > 0):
            raise ValueError("dim must be a positive integer")

        initial_distribution = LinearDiracDistribution(np.zeros((n_particles, dim)))
        AbstractParticleFilter.__init__(self, initial_distribution)
        AbstractEuclideanFilter.__init__(self, initial_distribution)

    @property
    def filter_state(self):
        """Get the filter state."""
        return self._filter_state

    @filter_state.setter
    def filter_state(
        self, new_state: Union[AbstractLinearDistribution, LinearDiracDistribution]
    ):
        """Set the filter state."""
        if not isinstance(new_state, LinearDiracDistribution):
            dist_dirac = LinearDiracDistribution.from_distribution(
                new_state, self._filter_state.d.shape[0]
            )
        else:
            dist_dirac = copy.deepcopy(new_state)

        if self._filter_state.d.shape != dist_dirac.d.shape:
            raise ValueError(
                "The shape of new state does not match with the existing state."
            )

        self._filter_state = dist_dirac

    @beartype
    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution: Optional[AbstractLinearDistribution] = None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = False,
    ):
        """Predict for nonlinear system model."""
        AbstractParticleFilter.predict_nonlinear(
            self, f, noise_distribution, function_is_vectorized, shift_instead_of_add
        )
