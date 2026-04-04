from collections.abc import Callable
from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    concatenate,
    hstack,
    int32,
    int64,
    mod,
    ones,
    pi,
    vstack,
    zeros,
)
from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import (
    HypercylindricalDiracDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import HypercylindricalFilterMixin


class HypercylindricalParticleFilter(AbstractParticleFilter, HypercylindricalFilterMixin):
    def __init__(
        self,
        n_particles: Union[int, int32, int64],
        bound_dim: Union[int, int32, int64],
        lin_dim: Union[int, int32, int64],
    ):
        d = zeros((n_particles, bound_dim + lin_dim))
        w = ones(n_particles) / n_particles
        filter_state = HypercylindricalDiracDistribution(bound_dim, d, w)
        HypercylindricalFilterMixin.__init__(self)
        AbstractParticleFilter.__init__(self, filter_state)

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
            d_f_applied = f(self.filter_state.d)
        else:
            self.filter_state.d = self.filter_state.apply_function(f).d
            d_f_applied = self.filter_state.d

        n_particles = self.filter_state.w.shape[0]
        if noise_distribution is None:
            updated_particles = d_f_applied
        else:
            updated_particles = []
            for i in range(n_particles):
                if not shift_instead_of_add:
                    noise = noise_distribution.sample(1)
                    updated_particles.append(d_f_applied[i] + noise)
                else:
                    noise_curr = noise_distribution.set_mean(d_f_applied[i])
                    updated_particles.append(noise_curr.sample(1))

            if self.filter_state.dim == 1:
                updated_particles = hstack(updated_particles)
            else:
                updated_particles = vstack(updated_particles)

        # Directly update particles to preserve bound_dim in the existing distribution
        self.filter_state.d = updated_particles

        # Wrap periodic dimensions to [0, 2*pi)
        bound_dim = self.filter_state.bound_dim
        wrapped_periodic = mod(self.filter_state.d[:, :bound_dim], 2.0 * pi)
        linear_part = self.filter_state.d[:, bound_dim:]
        self.filter_state.d = concatenate([wrapped_periodic, linear_part], axis=1)
