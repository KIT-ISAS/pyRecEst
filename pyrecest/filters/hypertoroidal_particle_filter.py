from pyrecest.backend import tile
from pyrecest.backend import sum
from pyrecest.backend import squeeze
from pyrecest.backend import mod
from pyrecest.backend import linspace
from pyrecest.backend import arange
from pyrecest.backend import int64
from pyrecest.backend import int32
from pyrecest.backend import zeros_like
import copy
from collections.abc import Callable

import numpy as np
from beartype import beartype
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    HypertoroidalDiracDistribution,
)
from pyrecest.distributions.circle.circular_dirac_distribution import (
    CircularDiracDistribution,
)

from .abstract_hypertoroidal_filter import AbstractHypertoroidalFilter
from .abstract_particle_filter import AbstractParticleFilter


class HypertoroidalParticleFilter(AbstractParticleFilter, AbstractHypertoroidalFilter):
    @beartype
    def __init__(
        self,
        n_particles: int | int32 | int64,
        dim: int | int32 | int64,
    ):
        assert np.isscalar(n_particles)
        assert n_particles > 1, "Use CircularParticleFilter for 1-D case"

        if dim == 1:
            # Prevents ambiguities if a vector is of size (dim,) or (n,) (for dim=1)
            filter_state = CircularDiracDistribution(
                linspace(0, 2 * np.pi, n_particles, endpoint=False)
            )
        else:
            filter_state = HypertoroidalDiracDistribution(
                tile(
                    linspace(0, 2 * np.pi, n_particles, endpoint=False), (dim, 1)
                ).T.squeeze(),
                dim=dim,
            )
        AbstractHypertoroidalFilter.__init__(self, filter_state)
        AbstractParticleFilter.__init__(self, filter_state)

    @beartype
    def set_state(self, new_state: AbstractHypertoroidalDistribution):
        if not isinstance(new_state, HypertoroidalDiracDistribution):
            # Convert to DiracDistribution if it is a different type of distribution
            # Use .__class__ to convert it to CircularDiracDistribution
            new_state = self.filter_state.__class__(
                new_state.sample(self.filter_state.w.size)
            )
        self.filter_state = copy.deepcopy(new_state)

    @beartype
    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution: AbstractHypertoroidalDistribution | None = None,
        function_is_vectorized: bool = True,
    ):
        if function_is_vectorized:
            self.filter_state.d = f(self.filter_state.d)
        else:
            self.filter_state.d = self.filter_state.apply_function(f)

        if noise_distribution is not None:
            noise = noise_distribution.sample(self.filter_state.w.size)
            self.filter_state.d += squeeze(noise)
            self.filter_state.d = mod(self.filter_state.d, 2 * np.pi)

    @beartype
    def predict_nonlinear_nonadditive(
        self, f: Callable, samples: np.ndarray, weights: np.ndarray
    ):
        assert (
            samples.shape[0] == weights.size
        ), "samples and weights must match in size"

        weights /= sum(weights)
        n = self.filter_state.shape[0]
        noise_ids = np.random.choice(arange(weights.size), size=n, p=weights)
        d = zeros_like(self.filter_state)
        for i in range(n):
            d[i, :] = f(self.filter_state[i, :], samples[noise_ids[i, :]])
        self.filter_state = d
