import numpy as np
from pyrecest.distributions import HypertoroidalDiracDistribution

from .abstract_hypertoroidal_filter import AbstractHypertoroidalFilter
from .abstract_particle_filter import AbstractParticleFilter


class HypertoroidalParticleFilter(AbstractParticleFilter, AbstractHypertoroidalFilter):
    def __init__(self, n_particles, dim):
        AbstractParticleFilter.__init__(self)
        self.filter_state = HypertoroidalDiracDistribution(
            np.tile(np.linspace(0, 2 * np.pi, n_particles, endpoint=False), (dim))
        )

    def set_state(self, state):
        if not isinstance(state, HypertoroidalDiracDistribution):
            state = HypertoroidalDiracDistribution(state.sample(self.filter_state.w.size))
        self.filter_state = state

    def predict_nonlinear(
        self, f, noise_distribution=None, function_is_vectorized=True
    ):
        if function_is_vectorized:
            self.filter_state = f(self.filter_state)
        else:
            self.filter_state = self.filter_state.apply_function(f)

        if noise_distribution is not None:
            noise = noise_distribution.sample(self.filter_state.w.size)
            self.filter_state.d += noise
            self.filter_state.d = np.mod(self.filter_state.d, 2 * np.pi)

    def predict_nonlinear_nonadditive(self, f, samples, weights):
        assert (
            samples.shape[0] == weights.size
        ), "samples and weights must match in size"
        assert callable(f), "f must be a function"

        weights /= np.sum(weights)
        n = self.filter_state.shape[0]
        noise_ids = np.random.choice(np.arange(weights.size), size=n, p=weights)
        d = np.zeros_like(self.filter_state)
        for i in range(n):
            d[i, :] = f(self.filter_state[i, :], samples[noise_ids[i, :]])
        self.filter_state = d
