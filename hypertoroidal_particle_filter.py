import numpy as np
from hypertoroidal_wd_distribution import HypertoroidalWDDistribution
from abstract_particle_filter import AbstractParticleFilter
from abstract_hypertoroidal_filter import AbstractHypertoroidalFilter

class HypertoroidalParticleFilter(AbstractParticleFilter, AbstractHypertoroidalFilter):
    def __init__(self, n_particles, dim):
        self.dist = HypertoroidalWDDistribution(np.tile(np.linspace(0, 2 * np.pi, n_particles, endpoint=False), (dim, 1)))

    def set_state(self, dist_):
        if not isinstance(dist_, HypertoroidalWDDistribution):
            dist_ = HypertoroidalWDDistribution(dist_.sample(self.dist.w.size))
        self.dist = dist_

    def predict_nonlinear(self, f, noise_distribution=None, function_is_vectorized=True):
        if function_is_vectorized:
            self.dist = f(self.dist)
        else:
            self.dist = self.dist.apply_function(f)

        if noise_distribution is not None:
            noise = noise_distribution.sample(self.dist.w.size)
            self.dist.d += noise
            self.dist.d = np.mod(self.dist.d, 2 * np.pi)

    def predict_nonlinear_nonadditive(self, f, samples, weights):
        assert samples.shape[1] == weights.size, "samples and weights must match in size"
        assert callable(f), "f must be a function"

        weights /= np.sum(weights)
        n = self.dist.shape[1]
        noise_ids = np.random.choice(np.arange(weights.size), size=n, p=weights)
        d = np.zeros_like(self.dist)
        for i in range(n):
            d[:, i] = f(self.dist[:, i], samples[:, noise_ids[i]])
        self.dist = d
