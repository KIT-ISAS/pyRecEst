from pyrecest.backend import random
from pyrecest.backend import sum
from pyrecest.backend import ones_like
from pyrecest.backend import ones
from pyrecest.backend import ndim
from collections.abc import Callable
from pyrecest.backend import zeros


from beartype import beartype
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)

from .abstract_filter_type import AbstractFilterType


class AbstractParticleFilter(AbstractFilterType):
    def __init__(self, initial_filter_state=None):
        AbstractFilterType.__init__(self, initial_filter_state)

    def predict_identity(self, noise_distribution):
        self.predict_nonlinear(f=lambda x: x, noise_distribution=noise_distribution)

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution=None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        assert (
            noise_distribution is None
            or self.filter_state.dim == noise_distribution.dim
        )

        if function_is_vectorized:
            self.filter_state.d = f(self.filter_state.d)
        else:
            self.filter_state = self.filter_state.apply_function(f)

        if noise_distribution is not None:
            if not shift_instead_of_add:
                noise = noise_distribution.sample(self.filter_state.w.shape[0])
                self.filter_state.d = self.filter_state.d + noise
            else:
                for i in range(self.filter_state.d.shape[1]):
                    noise_curr = noise_distribution.set_mean(self.filter_state.d[i, :])
                    self.filter_state.d[i, :] = noise_curr.sample(1)

    def predict_nonlinear_nonadditive(self, f, samples, weights):
        assert (
            samples.shape[0] == weights.shape[0]
        ), "samples and weights must match in size"

        weights = weights / sum(weights)
        n = self.filter_state.w.shape[0]
        noise_samples = random.choice(self.filter_state.d, n, p=weights)
        d = zeros((n, self.filter_state.dim))
        for i in range(n):
            d[i, :] = f(self.filter_state.d[i, :], noise_samples[i])

        self.filter_state.d = d

    def update_identity(
        self, meas_noise, measurement, shift_instead_of_add: bool = True
    ):
        assert measurement is None or measurement.shape == (meas_noise.dim,) or meas_noise.dim == 1 and measurement.shape == ()
        assert (
            ndim(measurement) == 1
            or ndim(measurement) == 0
            and meas_noise.dim == 1
        )
        if not shift_instead_of_add:
            raise NotImplementedError()

        likelihood = meas_noise.set_mode(measurement).pdf
        self.update_nonlinear_using_likelihood(likelihood)

    def update_nonlinear_using_likelihood(self, likelihood, measurement=None):
        if isinstance(likelihood, AbstractManifoldSpecificDistribution):
            likelihood = likelihood.pdf

        if measurement is None:
            self.filter_state = self.filter_state.reweigh(likelihood)
        else:
            self.filter_state = self.filter_state.reweigh(
                lambda x: likelihood(measurement, x)
            )

        self.filter_state.d = self.filter_state.sample(self.filter_state.w.shape[0])
        self.filter_state.w = (
            1 / self.filter_state.w.shape[0] * ones_like(self.filter_state.w)
        )

    def association_likelihood(self, likelihood: AbstractManifoldSpecificDistribution):
        likelihood_val = sum(
            likelihood.pdf(self.filter_state.d) * self.filter_state.w
        )
        return likelihood_val