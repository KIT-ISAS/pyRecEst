from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import float64, int32, int64, linspace, pi, sum
from pyrecest.distributions import CircularDiracDistribution

from .abstract_particle_filter import AbstractParticleFilter
from .hypertoroidal_particle_filter import HypertoroidalParticleFilter
from .manifold_mixins import CircularFilterMixin


class CircularParticleFilter(HypertoroidalParticleFilter, CircularFilterMixin):
    # pylint: disable=non-parent-init-called,super-init-not-called
    def __init__(self, n_particles: Union[int, int32, int64]) -> None:
        """
        Initialize the CircularParticleFilter.

        :param n_particles: number of particles
        """
        filter_state = CircularDiracDistribution(
            linspace(0.0, 2.0 * pi, n_particles, endpoint=False)
        )
        CircularFilterMixin.__init__(self)
        AbstractParticleFilter.__init__(self, filter_state)

    def compute_association_likelihood(self, likelihood) -> float64:
        """
        Compute the likelihood of association based on the PDF of the likelihood
        and the filter state.

        :param likelihood: likelihood object with a PDF method
        :return: association likelihood value
        """
        likelihood_val = sum(likelihood.pdf(self.filter_state.d) * self.filter_state.w)
        return likelihood_val
