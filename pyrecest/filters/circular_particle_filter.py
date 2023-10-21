from typing import Union
from pyrecest.backend import sum
from pyrecest.backend import float64
from pyrecest.backend import int64
from pyrecest.backend import int32


from .hypertoroidal_particle_filter import HypertoroidalParticleFilter


class CircularParticleFilter(HypertoroidalParticleFilter):
    def __init__(self, n_particles: Union[int, int32, int64]) -> None:
        """
        Initialize the CircularParticleFilter.

        :param n_particles: number of particles
        """
        super().__init__(n_particles, 1)

    def compute_association_likelihood(self, likelihood) -> float64:
        """
        Compute the likelihood of association based on the PDF of the likelihood
        and the filter state.

        :param likelihood: likelihood object with a PDF method
        :return: association likelihood value
        """
        likelihood_val = sum(
            likelihood.pdf(self.filter_state.d) * self.filter_state.w
        )
        return likelihood_val