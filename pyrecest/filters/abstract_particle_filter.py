import numpy as np
from pyrecest.distributions import AbstractDistribution
from .abstract_filter import AbstractFilter


class AbstractParticleFilter(AbstractFilter):
    def __init__(self):
        self.dist = None

    def set_state(self, dist_):
        assert isinstance(dist_, type(self.dist)), "New distribution has to be of the same class as (or inherit from) the previous density."
        self.dist = dist_

    def predict_identity(self, noise_distribution):
        self.predict_nonlinear(lambda x: x, noise_distribution)

    def predict_nonlinear(self, f, noise_distribution, function_is_vectorized=True, shift_instead_of_add=True):
        assert noise_distribution is None or self.dist.dim == noise_distribution.dim

        if function_is_vectorized:
            self.dist.d = f(self.dist.d)
        else:
            self.dist = self.dist.apply_function(f)

        if noise_distribution is not None:
            if not shift_instead_of_add:
                noise = noise_distribution.sample(self.dist.w.size)
                self.dist.d = self.dist.d + noise
            else:
                for i in range(self.dist.d.shape[1]):
                    noise_curr = noise_distribution.set_mode(self.dist.d[i, :])
                    self.dist.d[i, :] = noise_curr.sample(1)

    def predict_nonlinear_non_additive(self, f, samples, weights):
        assert samples.shape[0] == weights.size, 'samples and weights must match in size'

        weights = weights / np.sum(weights)
        n = self.dist.w.size
        noise_ids = np.random.choice(weights.size, n, p=weights)
        d = np.zeros((self.dist.dim, n))
        for i in range(n):
            d[i, :] = f(self.dist.d[i, :], samples[noise_ids[i]])

        self.dist.d = d

    def update_identity(self, noise_distribution, z, shift_instead_of_add=True):
        assert z is None or z.size == noise_distribution.dim
        if not shift_instead_of_add:
            raise NotImplementedError()
        else:
            noise_for_likelihood = noise_distribution.set_mode(z)
            likelihood = lambda x: noise_for_likelihood.pdf(x)
            self.update_nonlinear(likelihood)

    def update_nonlinear(self, likelihood, z=None):
        if isinstance(likelihood, AbstractDistribution):
            assert z is None, 'Cannot pass a density and a measurement. To assume additive noise, use update_identity.'
            likelihood = lambda x: likelihood.pdf(x)

        if z is None:
            self.dist = self.dist.reweigh(likelihood)
        else:
            self.dist = self.dist.reweigh(lambda x: likelihood(z, x))

        self.dist.d = self.dist.sample(self.dist.d.shape[0])
        self.dist.w = np.full(self.dist.d.shape[0], 1 / self.dist.d.shape[0])

    def get_estimate(self):
        return self.dist

    def association_likelihood(self, likelihood):
        likelihood_val = np.sum(likelihood.pdf(self.get_estimate().d) * self.get_estimate().w)
        return likelihood_val
