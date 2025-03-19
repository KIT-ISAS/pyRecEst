from collections.abc import Callable

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import ones  # noqa: F821
from pyrecest.distributions import AbstractHypersphericalDistribution
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter


class HyperhemisphereCartProdParticleFilter(AbstractParticleFilter):
    def __init__(
        self, n_particles: int | np.int32 | np.int64, dim_hemisphere, n_hemispheres
    ) -> None:
        """
        Constructor

        Parameters:
        n_particles (int > 0): Number of particles to use
        dim (int > 0): Dimension
        """
        initial_filter_state = HyperhemisphereCartProdDiracDistribution(
            np.empty((n_particles, (dim_hemisphere + 1) * n_hemispheres)),
            ones(n_particles) / n_particles,
            dim_hemisphere,
            n_hemispheres,
        )
        AbstractParticleFilter.__init__(self, initial_filter_state=initial_filter_state)

    def set_mean(self, new_mean):
        """
        Shifts all particles by the new mean, effectively changing the mean.
        """
        current_mean = self.filter_state.mean()
        shift = new_mean - current_mean
        self._filter_state.d += shift

    def set_state(self, new_state):
        """
        Sets the current system state

        Parameters:
        dist_ (HyperhemisphericalDiracDistribution): New state
        """
        assert isinstance(new_state, AbstractHyperhemisphericalDistribution)
        if not isinstance(new_state, HyperhemisphereCartProdDiracDistribution):
            new_state = HyperhemisphereCartProdDiracDistribution(
                new_state.sample(self.filter_state.d.shape[0]),
                w=ones(self.filter_state.d.shape[0]) / self.filter_state.d.shape[0],
                dim_hemisphere=self.filter_state.dim_hemisphere,
                n_hemispheres=self.filter_state.n_hemispheres,
            )
        self.filter_state = new_state

    def predict_nonlinear_each_part(
        self,
        f: Callable,
        noise_distribution: AbstractHyperhemisphericalDistribution,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        """
        Predicts the next state for each hyperhemisphere
        """
        assert function_is_vectorized, "Only vectorized functions are supported"
        assert (
            noise_distribution.dim == self.filter_state.dim_hemisphere
        ), "Noise dimension must match state dimension in Cartesian product"
        assert shift_instead_of_add, "Only shifting is supported"
        for i in range(self.filter_state.n_hemispheres):
            # Apply the function to each hyperhemisphere
            index_arr = range(
                i * (self.filter_state.dim_hemisphere + 1),
                (i + 1) * (self.filter_state.dim_hemisphere + 1),
            )
            # Consider only part of the state of the current hemisphere
            curr_d = self.filter_state.d[:, index_arr]
            d_fun_applied = f(curr_d)
            # Add noise
            if noise_distribution is not None:
                for j in range(self.filter_state.d.shape[0]):
                    # Set mean to transformed state
                    noise_distribution.set_mean(d_fun_applied[j])
                    # Sample one noise vector centered at the transformed state to add noise
                    self.filter_state.d[j, index_arr] = noise_distribution.sample(1)

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(
            new_state,
            (
                AbstractHyperhemisphericalDistribution,
                AbstractHypersphericalDistribution,
            ),
        ):
            assert new_state.dim == self.filter_state.dim_hemisphere
            samples = new_state.sample(
                self._filter_state.d.shape[0] * self._filter_state.n_hemispheres
            )
            if isinstance(new_state, AbstractHypersphericalDistribution):
                samples[samples[:, -1] < 0] = -samples[samples[:, -1] < 0]
            self._filter_state.d = samples.reshape(self.filter_state.d.shape)
        else:
            AbstractParticleFilter.filter_state.fset(self, new_state)
